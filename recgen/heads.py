import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, dropout: float):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _BaseHead(BaseEstimator):
    def __init__(
        self,
        hidden: list[int] = (256, 128),
        dropout: float = 0.1,
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int = 128,
        patience: int = 6,
        random_state: int = 0,
        device: str | None = None,
    ):
        self.hidden = list(hidden)
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    def _fit_nn(self, X, y, loss_fn, out_dim):
        torch.manual_seed(self.random_state)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        Xt = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
        yt = y
        if isinstance(yt, np.ndarray):
            yt = torch.tensor(yt, dtype=torch.float32).to(self.device)
        elif isinstance(yt, (list, tuple)):
            yt = torch.tensor(yt, dtype=torch.float32).to(self.device)
        model = _MLP(X.shape[1], self.hidden, out_dim, self.dropout).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        n = len(Xt)
        split = int(n * 0.85)
        Xtr, Xva = Xt[:split], Xt[split:]
        ytr, yva = yt[:split], yt[split:]
        best = np.inf
        best_state = None
        patience_left = self.patience
        for epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(self.random_state + epoch))
            for i in range(0, len(Xtr), self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vloss = loss_fn(model(Xva), yva).item()
            if vloss < best - 1e-5:
                best = vloss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        self.model_.eval()

    def _predict_tensor(self, X):
        Xs = self.scaler_.transform(X)
        Xt = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
        with torch.no_grad():
            return self.model_(Xt)


class ClassificationHead(_BaseHead, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y))
        y_idx = np.searchsorted(self.classes_, y)
        yt = torch.tensor(y_idx, dtype=torch.long).to(self.device)
        self._fit_nn(X, yt, nn.CrossEntropyLoss(), len(self.classes_))
        return self

    def predict_proba(self, X):
        logits = self._predict_tensor(X)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[probs.argmax(axis=1)]


class RegressionHead(_BaseHead, RegressorMixin):
    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float32)
        self._fit_nn(X, y.reshape(-1, 1), nn.MSELoss(), 1)
        return self

    def predict(self, X):
        return self._predict_tensor(X).cpu().numpy().ravel()
