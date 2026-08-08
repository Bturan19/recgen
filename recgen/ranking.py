import numpy as np
import torch
import torch.nn as nn

from .heads import _BaseHead


class CatalogRankingHead(_BaseHead):
    """GenRec-style catalog-aware head: score(u,i) = <W h_u, e_i> + item_bias.
    Item embeddings e_i are frozen LLM encodings of item text."""

    def __init__(
        self,
        dim: int = 960,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 256,
        patience: int = 5,
        random_state: int = 0,
        device: str | None = None,
    ):
        self.dim = dim
        super().__init__(
            hidden=(),
            dropout=0.0,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            random_state=random_state,
            device=device,
        )

    def fit(self, H, y, E):
        """H: (n_users, dim) frozen LLM embeddings of histories.
        y: (n_users,) integer item ids in [0, n_items).
        E: (n_items, dim) frozen LLM embeddings of catalog items."""
        torch.manual_seed(self.random_state)
        self.n_items_ = E.shape[0]
        self.proj_ = nn.Linear(self.dim, self.dim).to(self.device)
        self.item_bias_ = torch.zeros(self.n_items_, device=self.device)
        init_logits = torch.tensor(H @ E.T, dtype=torch.float64)
        self.temp_init_ = 1.0 / max(float(torch.std(init_logits)), 1e-3)
        self.temp_ = nn.Parameter(torch.tensor(self.temp_init_))

        Ht = torch.from_numpy(H.astype(np.float32)).to(self.device)
        Et = torch.from_numpy(E.astype(np.float32)).to(self.device)
        yt = torch.tensor(y, dtype=torch.long).to(self.device)

        n = len(Ht)
        split = int(n * 0.9)
        Htr, Hva = Ht[:split], Ht[split:]
        ytr, yva = yt[:split], yt[split:]

        opt = torch.optim.AdamW(
            [{"params": [self.temp_, self.item_bias_], "lr": self.lr},
             {"params": self.proj_.parameters(), "lr": self.lr * 0.5}],
            weight_decay=1e-4,
        )
        best = -np.inf
        best_state = None
        patience_left = self.patience
        for epoch in range(self.epochs):
            self.proj_.train()
            perm = torch.randperm(len(Htr), generator=torch.Generator().manual_seed(self.random_state + epoch))
            for i in range(0, len(Htr), self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = self._loss(Htr[idx], ytr[idx], Et)
                loss.backward()
                opt.step()
            self.proj_.eval()
            with torch.no_grad():
                mrr = self._mrr(Hva, yva, Et)
            if mrr > best:
                best = mrr
                best_state = {
                    "temp": self.temp_.detach().clone(),
                    "item_bias": self.item_bias_.detach().clone(),
                    "proj": {k: v.clone() for k, v in self.proj_.state_dict().items()},
                }
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            self.temp_ = nn.Parameter(best_state["temp"])
            self.item_bias_ = best_state["item_bias"]
            self.proj_.load_state_dict(best_state["proj"])
        self.E_ = Et
        self.best_mrr_ = best
        return self

    def _loss(self, H, y, E):
        logits = self._logits(H, E)
        return torch.nn.functional.cross_entropy(logits, y)

    def _logits(self, H, E):
        h = self.proj_(H)
        logits = self.temp_ * (h @ E.T)
        logits = logits + self.item_bias_.unsqueeze(0)
        return logits

    def _mrr(self, H, y, E):
        logits = self._logits(H, E)
        ranks = torch.argsort(logits, dim=-1, descending=True)
        pos = (ranks == y.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
        return float(torch.mean(1.0 / pos))

    def predict_scores(self, H, E=None) -> np.ndarray:
        E = E if E is not None else self.E_
        if isinstance(H, torch.Tensor):
            Ht = H.float()
        else:
            Ht = torch.from_numpy(np.asarray(H, dtype=np.float32)).to(self.device)
        if isinstance(E, torch.Tensor):
            Et = E.float()
        else:
            Et = torch.from_numpy(np.asarray(E, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            return self._logits(Ht, Et).cpu().numpy()

    def evaluate(self, H, y, ks=(10, 20), E=None):
        scores = self.predict_scores(H, E)
        order = np.argsort(-scores, axis=1)
        ranks = np.zeros(len(y), dtype=int)
        for i, o in enumerate(order):
            ranks[i] = np.where(o == y[i])[0][0] + 1
        out = {"mrr@20": float(np.mean(1.0 / ranks))}
        for k in ks:
            out[f"recall@{k}"] = float(np.mean(ranks <= k))
        return out
