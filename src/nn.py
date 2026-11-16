"""A small feedforward neural network in plain NumPy (Lecture 4).

Written out by hand, instead of calling a deep-learning library, so every step from
the lecture is visible and checkable:

* hidden layers  a = ReLU(W x + b)                       (Lecture 4, slides 33-39)
* output         sigmoid + log loss for fraud / not fraud (binary label)
                 identity + squared error for the autoencoder (reconstruct the input)
* training       feedforward -> loss -> backpropagation -> parameter update, repeated
                 over shuffled mini-batches; one pass over the data = one epoch
* optimiser      mini-batch gradient descent with Adam step sizes
* regularisation dropout on hidden units during training; at inference all units are
                 used and activations are multiplied by (1 - dropout rate)
                 ("Method 1", Lecture 4 slide 71), plus an optional L2 penalty (Lecture 3)
* early stopping keep the weights from the epoch with the best validation score

The same class is used as a supervised classifier (MLP) and, with output="identity"
and a narrow middle layer, as an autoencoder trained on legitimate applications only.
"""
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class MLP:
    def __init__(self, hidden=(64, 32), output="sigmoid", dropout=0.0, lr=1e-3, batch_size=512,
                 epochs=20, l2=0.0, patience=3, seed=0):
        if output not in ("sigmoid", "identity"):
            raise ValueError("output must be 'sigmoid' (classifier) or 'identity' (autoencoder)")
        self.hidden, self.output, self.dropout = tuple(hidden), output, float(dropout)
        self.lr, self.batch_size, self.epochs = lr, batch_size, epochs
        self.l2, self.patience, self.seed = l2, patience, seed
        self.history = []

    # ---- forward / backward -------------------------------------------------
    def _init(self, n_in, n_out):
        rng = np.random.default_rng(self.seed)
        sizes = [n_in, *self.hidden, n_out]
        # He initialisation suits ReLU layers
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self._m = [np.zeros_like(p) for p in self.W + self.b]   # Adam first moment
        self._v = [np.zeros_like(p) for p in self.W + self.b]   # Adam second moment
        self._t = 0

    def _forward(self, X, train, rng=None):
        acts, masks, a = [X], [], X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            a = np.maximum(0.0, a @ W + b)                       # ReLU hidden layer
            if self.dropout > 0:
                if train:                                        # drop each unit with prob p
                    m = (rng.random(a.shape) >= self.dropout).astype(a.dtype)
                    a = a * m
                    masks.append(m)
                else:                                            # all units on, scaled by (1 - p)
                    a = a * (1.0 - self.dropout)
            acts.append(a)
        z = a @ self.W[-1] + self.b[-1]
        out = _sigmoid(z) if self.output == "sigmoid" else z
        return out, acts, masks

    def _loss(self, out, y):
        if self.output == "sigmoid":
            p = np.clip(out[:, 0], 1e-7, 1 - 1e-7)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return float(np.mean((out - y) ** 2))

    def _backward(self, X, y, rng):
        """Feedforward (with dropout) then backpropagation; returns loss and gradients."""
        out, acts, masks = self._forward(X, train=True, rng=rng)
        n = len(X)
        # error signal at the output: for sigmoid + log loss and for identity + squared
        # error the gradient w.r.t. the pre-activation is (prediction - label) / n
        delta = (out - y[:, None]) / n if self.output == "sigmoid" else 2 * (out - y) / out.size
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.b)
        for layer in range(len(self.W) - 1, -1, -1):
            grads_W[layer] = acts[layer].T @ delta + self.l2 * self.W[layer]
            grads_b[layer] = delta.sum(axis=0)
            if layer > 0:                                        # backpropagate to the layer below
                delta = (delta @ self.W[layer].T) * (acts[layer] > 0)
                if self.dropout > 0:
                    delta = delta * masks[layer - 1]
        return self._loss(out, y), grads_W, grads_b

    def _step(self, X, y, rng):
        loss, grads_W, grads_b = self._backward(X, y, rng)
        # Adam update
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params, grads = self.W + self.b, grads_W + grads_b
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g * g
            m_hat = self._m[i] / (1 - b1 ** self._t)
            v_hat = self._v[i] / (1 - b2 ** self._t)
            p -= (self.lr * m_hat / (np.sqrt(v_hat) + eps)).astype(p.dtype)
        return loss

    # ---- public API -----------------------------------------------------------
    def fit(self, X, y=None, X_val=None, y_val=None, val_score=None):
        """Train with mini-batches. Autoencoder: pass y=None (the target is X itself).

        val_score(model) -> float, higher is better; used for early stopping when given,
        otherwise the validation loss is used (lower is better)."""
        X = np.asarray(X, dtype=np.float32)
        target = X if y is None else np.asarray(y, dtype=np.float32)
        self._init(X.shape[1], 1 if self.output == "sigmoid" else X.shape[1])
        rng = np.random.default_rng(self.seed)
        best, best_params, bad = -np.inf, None, 0
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(len(X))
            losses = [self._step(X[idx], target[idx], rng)
                      for idx in np.array_split(order, max(1, len(X) // self.batch_size))]
            rec = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            if val_score is not None:
                rec["val_score"] = float(val_score(self))
            elif X_val is not None:
                Xv = np.asarray(X_val, dtype=np.float32)
                tv = Xv if y_val is None else np.asarray(y_val, dtype=np.float32)
                rec["val_score"] = -self._loss(self._forward(Xv, train=False)[0], tv)
            self.history.append(rec)
            if "val_score" not in rec:
                continue
            if rec["val_score"] > best + 1e-6:
                best, bad = rec["val_score"], 0
                best_params = ([w.copy() for w in self.W], [b.copy() for b in self.b], epoch)
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_params is not None:
            self.W, self.b, self.best_epoch = best_params
        else:
            self.best_epoch = len(self.history)
        return self

    def predict_proba(self, X):
        """Fraud score in (0, 1) for the classifier (column 1, sklearn style)."""
        p = self._forward(np.asarray(X, dtype=np.float32), train=False)[0][:, 0]
        return np.column_stack([1 - p, p])

    def reconstruct(self, X):
        return self._forward(np.asarray(X, dtype=np.float32), train=False)[0]

    def reconstruction_error(self, X, batch=100_000):
        """Mean squared reconstruction error per row = anomaly score (autoencoder)."""
        X = np.asarray(X, dtype=np.float32)
        return np.concatenate([np.mean((self.reconstruct(X[i:i + batch]) - X[i:i + batch]) ** 2, axis=1)
                               for i in range(0, len(X), batch)])
