"""Marketplace moderation VLM distillation (Qwen2.5-VL-3B + LoRA on MPS).

Fine-tunes the VLM to predict the eval-labeler's decisions (distillation),
so it learns the labeler's judgment style instead of judging from first
principles. Output is decision-only (1 token at inference) so serving is
fast. Vision tower frozen; LoRA on the language model.

Run: uv run python experiments/moderation/vlm_lora.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from data import image_paths, load, stratified_split

MODEL_DIR = "models/Qwen2.5-VL-3B"
OUT_DIR = "checkpoints/moderation_vlm"
MAX_IMGS = 1
IMG_SIZE = 224
EPOCHS = 1
LR = 1e-4
BATCH = 2
ACCUM = 2
LORA_R = 8

SYS = (
    "Sen bir e-ticaret moderasyon uzmanısın. Ürün görsellerini ve bilgilerini incele. "
    "Reddet: marka uyumsuzluğu (görseldeki marka bilgidekinden farklıysa), "
    "başlık/resim/açıklama uyuşmazlığı, cinsellik, sağlık beyanı, yasa dışı madde, "
    "yanıltıcı ürün. Aksi halde onayla. Sadece kararı yaz: Onaylandı veya Reddedildi"
)


def build_example(r, tag):
    from PIL import Image

    imgs = [Image.open(p).convert("RGB") for p in image_paths(r["ProductId"], MAX_IMGS)]
    if not imgs:
        imgs = [Image.new("RGB", (16, 16), "white")]
    user_text = (
        f"Ürün başlığı: {r['DisplayName']}\nMarka: {r['BrandName']}\n"
        f"Kategori: {r['CategoryHierarchy']}\nAçıklama: {(r['Description'] or '')[:200]}"
    )
    ans = f"Reddedildi: {tag}" if tag else "Onaylandı"
    return imgs, user_text, ans


def _build_msgs(indices, rows, y, tags, with_answer):
    from PIL import Image

    msgs, y_l = [], []
    for i in indices:
        r = rows[i]
        tag = (tags[i][0] if tags[i] and y[i] == 1 else None)
        imgs, ut, ans = build_example(r, tag)
        content = [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": ut}]
        m = [{"role": "system", "content": SYS}, {"role": "user", "content": content}]
        if with_answer:
            m.append({"role": "assistant", "content": ans})
        msgs.append(m)
        y_l.append(y[i])
    return msgs, np.array(y_l)


def _tokenize(msgs, proc, add_generation_prompt):
    texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=add_generation_prompt) for m in msgs]
    images = [[c["image"] for c in m[1]["content"] if c["type"] == "image"] for m in msgs]
    return proc(text=texts, images=images, return_tensors="pt", padding=True).to("mps")


def collate(indices, rows, y, tags, proc):
    msgs, y_l = _build_msgs(indices, rows, y, tags, with_answer=True)
    inputs = _tokenize(msgs, proc, add_generation_prompt=False)
    ans_counts = []
    for i in indices:
        r = rows[i]
        tag = (tags[i][0] if tags[i] and y[i] == 1 else None)
        ans = f"Reddedildi: {tag}" if tag else "Onaylandı"
        ans_counts.append(len(proc.tokenizer(ans, add_special_tokens=False)["input_ids"]))
    return inputs, y_l, ans_counts


def collate_eval(indices, rows, y, tags, proc):
    msgs, y_l = _build_msgs(indices, rows, y, tags, with_answer=False)
    inputs = _tokenize(msgs, proc, add_generation_prompt=True)
    return inputs, y_l


def main():
    from PIL import Image

    df = load()
    rows = df.to_dicts()
    y = df["eval_decision"].to_numpy()
    tags = df["eval_rejection_tag"].to_list()
    tr, va, te = stratified_split(df)
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

    print("loading model...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.float16, torch_dtype=torch.float16).to("mps")
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    for name, p in model.named_parameters():
        if name.startswith("model.visual."):
            p.requires_grad = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        ),
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable:,}", flush=True)
    model.train()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    rng = np.random.default_rng(0)

    best_acc, best_state, patience = -1.0, None, 0
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        perm = rng.permutation(len(tr))
        total = 0
        correct = 0
        for i in range(0, len(perm), BATCH):
            b = perm[i : i + BATCH].tolist()
            inputs, yb, ans_counts = collate(b, rows, y, tags, proc)
            labels = inputs["input_ids"].clone()
            labels.fill_(-100)
            for k, n_ans in enumerate(ans_counts):
                labels[k, -n_ans:] = inputs["input_ids"][k, -n_ans:]
            try:
                out = model(**inputs, labels=labels)
            except Exception as e:
                print(f"  !! step error {type(e).__name__}, skipping", flush=True)
                opt.zero_grad()
                continue
            loss = out.loss / ACCUM
            if not torch.isfinite(loss):
                print(f"  !! NaN loss at step {i // BATCH}, skipping", flush=True)
                opt.zero_grad()
                continue
            loss.backward()
            if (i // BATCH + 1) % ACCUM == 0 or i + BATCH >= len(perm):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
            if (i // BATCH) % 20 == 0:
                print(f"  ep{epoch + 1} {i + len(b)}/{len(perm)} loss={loss.item() * ACCUM:.3f} ({time.time() - t0:.0f}s)", flush=True)
            if (i // BATCH) % 300 == 0 and (i // BATCH) > 0:
                model.save_pretrained(OUT_DIR + "_ckpt")

        # val accuracy: 1-token decision (Onaylandı / Reddedildi)
        model.eval()
        va_correct = 0
        with torch.no_grad():
            for i in range(0, len(va), BATCH):
                b = va[i : i + BATCH].tolist()
                inputs, yb, _ = collate(b, rows, y, tags, proc)
                inputs.pop("labels", None)
                gen = model.generate(**inputs, max_new_tokens=6, do_sample=False)
                for k, idx in enumerate(b):
                    ans = proc.decode(gen[k][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                    pred = 1 if "Reddedildi" in ans else 0
                    va_correct += pred == y[k]
        vacc = va_correct / len(va)
        print(f"epoch {epoch + 1}: val acc={vacc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vacc > best_acc:
            best_acc = vacc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 0:
                break

    model.load_state_dict(best_state)
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    print(f"saved -> {OUT_DIR} (best val acc {best_acc:.4f})")

    # test evaluation
    model.eval()
    te_correct = 0
    preds = []
    with torch.no_grad():
        for i in range(0, len(te), BATCH):
            b = te[i : i + BATCH].tolist()
            inputs, yb, _ = collate(b, rows, y, tags, proc)
            gen = model.generate(**inputs, max_new_tokens=6, do_sample=False)
            for k, idx in enumerate(b):
                ans = proc.decode(gen[k][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                pred = 1 if "Reddedildi" in ans else 0
                preds.append(pred)
                te_correct += pred == y[idx]
    acc = te_correct / len(te)
    print(f"TEST acc={acc:.4f}")
    np.save(os.path.join(OUT_DIR, "test_preds.npy"), np.array(preds))


if __name__ == "__main__":
    main()
