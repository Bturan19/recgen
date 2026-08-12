import os
import time

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .heads import _MLP


class LoraHeadTrainer:
    def __init__(
        self,
        encoder,
        task: str = "classifier",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        head_hidden: tuple[int, ...] = (256, 128),
        head_dropout: float = 0.1,
        lr_lora: float = 2e-4,
        lr_head: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 8,
        grad_accum: int = 4,
        patience: int = 2,
        max_length: int = 512,
        seed: int = 0,
    ):
        self.encoder = encoder
        self.task = task
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = list(lora_target_modules)
        self.head_hidden = list(head_hidden)
        self.head_dropout = head_dropout
        self.lr_lora = lr_lora
        self.lr_head = lr_head
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.patience = patience
        self.max_length = max_length
        self.seed = seed
        self.device = encoder.device

    def _prepare(self, X, y):
        torch.manual_seed(self.seed)
        self.verbalizer = getattr(self, "verbalizer", None)
        if self.verbalizer is not None:
            self.verbalizer.fit(X)
            texts = self.verbalizer.transform(X)
        else:
            texts = list(X)
        y = np.asarray(y)
        n = len(texts)
        split = int(n * 0.8)
        idx = np.random.default_rng(self.seed).permutation(n)
        tr_idx, va_idx = idx[:split], idx[split:]
        self._texts_tr = [texts[i] for i in tr_idx]
        self._texts_va = [texts[i] for i in va_idx]
        self._y_tr = y[tr_idx]
        self._y_va = y[va_idx]
        if self.task == "classifier":
            self.classes_ = np.unique(y)
            self._y_tr_idx = np.searchsorted(self.classes_, self._y_tr)
            self._y_va_idx = np.searchsorted(self.classes_, self._y_va)

    def _collate(self, texts):
        ids = self.encoder.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return ids["input_ids"].to(self.device), ids["attention_mask"].to(self.device)

    def _pool(self, hidden, attn):
        from .encoder import pool_hidden

        return pool_hidden(hidden, attn, self.encoder.pooling).contiguous()

    def _encode_batch(self, input_ids, attn):
        out = self.lora_model.base_model.model.model(input_ids=input_ids, attention_mask=attn)
        hidden = out.last_hidden_state.float()
        return self._pool(hidden, attn).contiguous()

    def fit(self, X, y, verbalizer=None, out_dir="checkpoints"):
        self.verbalizer = verbalizer
        self._prepare(X, y)
        print(f"[LoraHeadTrainer] train={len(self._texts_tr)} val={len(self._texts_va)} task={self.task}")

        if next(self.encoder.model.parameters()).dtype == torch.float16:
            self.encoder.model.float()
            print("[LoraHeadTrainer] converted backbone to fp32 for stable MPS autograd")
        self.encoder.model.config.use_cache = False
        self.lora_model = get_peft_model(
            self.encoder.model,
            LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        out_dim = len(self.classes_) if self.task == "classifier" else 1
        self.head = _MLP(self.encoder.dim, self.head_hidden, out_dim, self.head_dropout).to(self.device)

        lora_params = [p for p in self.lora_model.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        n_lora = sum(p.numel() for p in lora_params)
        print(f"[LoraHeadTrainer] LoRA params: {n_lora:,}, head params: {sum(p.numel() for p in head_params):,}")
        opt = torch.optim.AdamW(
            [{"params": lora_params, "lr": self.lr_lora}, {"params": head_params, "lr": self.lr_head}],
            weight_decay=1e-4,
        )

        ds_tr = TensorDataset(
            *self._collate(self._texts_tr),
            torch.tensor(self._y_tr_idx if self.task == "classifier" else self._y_tr, dtype=torch.long if self.task == "classifier" else torch.float32).to(self.device),
        )
        loader = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True)
        loss_fn = nn.CrossEntropyLoss() if self.task == "classifier" else nn.MSELoss()

        best = -np.inf if self.task == "classifier" else np.inf
        best_state = None
        patience_left = self.patience
        t0 = time.time()
        for epoch in range(self.epochs):
            self.lora_model.train()
            self.head.train()
            opt.zero_grad()
            for i, (ids, attn, yb) in enumerate(loader):
                h = self._encode_batch(ids, attn)
                logits = self.head(h)
                if self.task == "classifier":
                    loss = loss_fn(logits, yb)
                else:
                    loss = loss_fn(logits.squeeze(-1), yb)
                loss = loss / self.grad_accum
                loss.backward()
                if (i + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
                    opt.step()
                    opt.zero_grad()
            self.lora_model.eval()
            self.head.eval()
            metric = self._eval_val()
            tag = "auc" if self.task == "classifier" else "mae"
            print(f"  epoch {epoch + 1}/{self.epochs} val_{tag}={metric:.4f} ({time.time() - t0:.0f}s)")
            better = metric > best if self.task == "classifier" else metric < best
            if better:
                best = metric
                best_state = {
                    "lora": {k: v.clone() for k, v in self.lora_model.state_dict().items()},
                    "head": {k: v.clone() for k, v in self.head.state_dict().items()},
                }
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"  early stop at epoch {epoch + 1}")
                    break

        self.lora_model.load_state_dict(best_state["lora"])
        self.head.load_state_dict(best_state["head"])
        self._save(out_dir)
        print(f"[LoraHeadTrainer] done, best val_{tag}={best:.4f}")
        return self

    def _eval_val(self):
        with torch.no_grad():
            h = self._encode_batch(*self._collate(self._texts_va))
            logits = self.head(h)
        if self.task == "classifier":
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
            return roc_auc_score(self._y_va, proba[:, 1])
        pred = logits.squeeze(-1).cpu().numpy()
        return mean_absolute_error(self._y_va, pred)

    def load(self, out_dir: str, X=None, y=None, verbalizer=None):
        from peft import PeftModel

        self.verbalizer = verbalizer
        self.classes_ = np.load(f"{out_dir}/classes.npy")
        out_dim = len(self.classes_) if self.task == "classifier" else 1
        self.head = _MLP(self.encoder.dim, self.head_hidden, out_dim, self.head_dropout).to(self.device)
        self.head.load_state_dict(torch.load(f"{out_dir}/head.pt", map_location=self.device))
        self.lora_model = PeftModel.from_pretrained(self.encoder.model, f"{out_dir}/adapter")
        self.lora_model.eval()
        self.head.eval()
        if X is not None:
            self._prepare(X, y)
        return self

    def _save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.head.state_dict(), f"{out_dir}/head.pt")
        np.save(f"{out_dir}/classes.npy", self.classes_)
        adapter_dir = f"{out_dir}/adapter"
        self.lora_model.save_pretrained(adapter_dir)
        print(f"[LoraHeadTrainer] saved -> {out_dir}")

    def predict_proba(self, X):
        texts = self._texts(X)
        self.lora_model.eval()
        self.head.eval()
        h = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size * 4):
                ids, attn = self._collate(texts[i : i + self.batch_size * 4])
                h.append(self._encode_batch(ids, attn).cpu())
        H = torch.cat(h)
        logits = self.head(H.to(self.device))
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()

    def predict(self, X):
        if self.task == "classifier":
            probs = self.predict_proba(X)
            return self.classes_[probs.argmax(axis=1)]
        texts = self._texts(X)
        self.lora_model.eval()
        self.head.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size * 4):
                ids, attn = self._collate(texts[i : i + self.batch_size * 4])
                preds.append(self.head(self._encode_batch(ids, attn)).squeeze(-1).cpu())
        return torch.cat(preds).numpy()

    def _texts(self, X):
        if self.verbalizer is not None:
            return self.verbalizer.transform(X)
        return list(X)
