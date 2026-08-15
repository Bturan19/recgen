"""Marketplace moderation — full fine-tune of a tiny LLM as a generative judge.

The task is narrow, so a small model with all parameters tuned to it often
beats a large frozen encoder + head (fewer parameters = less memorization,
better pattern learning on 4k rows). Generative output (Onaylandı /
Reddedildi) makes serving 1-2 tokens.

Run: uv run python experiments/moderation/tiny_finetune.py --model qwen05
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load, stratified_split

MODELS = {
    "smol135": "models/SmolLM2-135M",
    "qwen05": "models/Qwen2.5-0.5B",
}

SYS = (
    "Sen bir e-ticaret moderasyon uzmanısın. Ürün bilgilerini incele. "
    "Reddet: marka uyumsuzluğu, başlık/açıklama uyuşmazlığı, cinsellik, sağlık beyanı, "
    "yasa dışı madde, yanıltıcı ürün, iletişim bilgisi. Aksi halde onayla."
    "Sadece 'Onaylandı' veya 'Reddedildi' yaz."
)

ANSWER_TOK = {"Onaylandı": 0, "Reddedildi": 1}


def build_text(r):
    return (
        f"Ürün başlığı: {r['DisplayName']}\n"
        f"Marka: {r['BrandName']}\n"
        f"Kategori: {r['CategoryHierarchy']}\n"
        f"Öznitelikler: {r['AttributesJson']}\n"
        f"Açıklama: {(r['Description'] or '')[:350]}"
    )


def main(model_key: str = "qwen05", epochs: int = 6, batch: int = 8, lr: float = 3e-4):
    df = load()
    rows = df.to_dicts()
    y = df["eval_decision"].to_numpy()
    tags = df["eval_rejection_tag"].to_list()
    tr, va, te = stratified_split(df)

    tok = AutoTokenizer.from_pretrained(MODELS[model_key])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODELS[model_key]).to("mps")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{model_key}: {n_params/1e6:.0f}M params, all trainable", flush=True)

    def build_batch(indices, with_answer):
        texts, labels_list = [], []
        for i in indices:
            r = rows[i]
            tag = (tags[i][0] if tags[i] and y[i] == 1 else None)
            ans = f"Reddedildi: {tag}" if tag else "Onaylandı"
            prompt = build_text(r) + "\nKarar:"
            texts.append(prompt + (ans if with_answer else ""))
            labels_list.append(ans)
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("mps")
        if with_answer:
            ans_counts = [len(tok(a, add_special_tokens=False)["input_ids"]) for a in labels_list]
            labels = enc["input_ids"].clone()
            labels.fill_(-100)
            for k, n in enumerate(ans_counts):
                labels[k, -n:] = enc["input_ids"][k, -n:]
        else:
            labels = None
        return enc, labels

    rng = np.random.default_rng(0)
    t0 = time.time()
    best_acc, best_state, patience = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch].tolist()
            enc, labels = build_batch(b, with_answer=True)
            out = model(**enc, labels=labels)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (i // batch) % 25 == 0:
                print(f"  ep{epoch+1} {i+len(b)}/{len(perm)} loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
        # val: generate 1-3 tokens, check decision
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(va), batch):
                b = va[i : i + batch].tolist()
                enc, _ = build_batch(b, with_answer=False)
                gen = model.generate(**enc, max_new_tokens=8, do_sample=False)
                for k, idx in enumerate(b):
                    ans = tok.decode(gen[k][enc["input_ids"].shape[1] :], skip_special_tokens=True)
                    pred = 1 if "Reddedildi" in ans else 0
                    correct += pred == y[idx]
        vacc = correct / len(va)
        print(f"epoch {epoch+1}: val acc={vacc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vacc > best_acc:
            best_acc = vacc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    model.load_state_dict(best_state)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(best_state, f"checkpoints/tiny_{model_key}.pt")
    print(f"saved checkpoints/tiny_{model_key}.pt (best val {best_acc:.4f})", flush=True)

    # test
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(te), batch):
            b = te[i : i + batch].tolist()
            enc, _ = build_batch(b, with_answer=False)
            gen = model.generate(**enc, max_new_tokens=8, do_sample=False)
            for k, idx in enumerate(b):
                ans = tok.decode(gen[k][enc["input_ids"].shape[1] :], skip_special_tokens=True)
                pred = 1 if "Reddedildi" in ans else 0
                correct += pred == y[idx]
    print(f"TEST acc={correct/len(te):.4f} ({correct}/{len(te)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen05", choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()
    main(model_key=args.model, epochs=args.epochs)
