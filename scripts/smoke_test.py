"""Smoke test: load SmolLM2-360M on MPS, encode two verbalized rows, compare embeddings."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "models/SmolLM2-360M"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading tokenizer + model from {MODEL_DIR} on {DEVICE}...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to(DEVICE)
model.eval()

rows = [
    "Age: 39 | Workclass: State-gov | Education: Bachelors | Education-num: 13 | Marital-status: Never-married | Occupation: Adm-clerical | Relationship: Not-in-family | Race: White | Sex: Male | Capital-gain: 2174 | Capital-loss: 0 | Hours-per-week: 40 | Native-country: United-States | Income: >50K",
    "Age: 50 | Workclass: Self-emp-not-inc | Education: Bachelors | Education-num: 13 | Marital-status: Married-civ-spouse | Occupation: Exec-managerial | Relationship: Husband | Race: White | Sex: Male | Capital-gain: 0 | Capital-loss: 0 | Hours-per-week: 13 | Native-country: United-States | Income: >50K",
    "Age: 28 | Workclass: Private | Education: HS-grad | Education-num: 9 | Marital-status: Divorced | Occupation: Sales | Relationship: Not-in-family | Race: Black | Sex: Female | Capital-gain: 0 | Capital-loss: 0 | Hours-per-week: 30 | Native-country: Mexico | Income: <=50K",
]


def encode(prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    hidden = out.hidden_states[-1]  # (1, seq_len, 960)
    seq = hidden[0]
    last_tok = seq[-1]
    mean = seq.mean(dim=0)
    return last_tok, mean


t0 = time.time()
lasts, means = [], []
for r in rows:
    l, m = encode(r)
    lasts.append(l)
    means.append(m)
dt = time.time() - t0

lasts = torch.stack(lasts)
means = torch.stack(means)
lasts = torch.nn.functional.normalize(lasts, dim=-1)
means = torch.nn.functional.normalize(means, dim=-1)

print(f"\nEncoded {len(rows)} prompts in {dt:.1f}s ({len(rows)/dt:.2f} prompts/s)")
print(f"Embedding dim: {lasts.shape[1]}")
print("\nCosine sims (last-token / mean-pool):")
print(f"  row1 vs row2 (both >50K): {torch.dot(lasts[0], lasts[1]).item():.4f} / {torch.dot(means[0], means[1]).item():.4f}")
print(f"  row1 vs row3 (opposite):   {torch.dot(lasts[0], lasts[2]).item():.4f} / {torch.dot(means[0], means[2]).item():.4f}")
print(f"  row2 vs row3 (opposite):   {torch.dot(lasts[1], lasts[2]).item():.4f} / {torch.dot(means[1], means[2]).item():.4f}")
print("\nSMOKE TEST OK")
