"""Checks for the hand-written network in src/nn.py.

    python -m tests.test_nn

1. Backpropagation matches a numerical (finite-difference) gradient.
2. Dropout at inference scales activations by (1 - p), so the expected pre-activation
   matches training (Lecture 4, slide 71).
3. The classifier learns an easy separable problem; the autoencoder reconstructs
   normal rows better than shifted (anomalous) rows.
"""
import numpy as np

from src.nn import MLP


def _numeric_grad_check(output):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4)).astype(np.float64)
    m = MLP(hidden=(5, 3), output=output, dropout=0.0, lr=0.0, seed=1)
    m._init(4, 1 if output == "sigmoid" else 4)
    m.W = [w.astype(np.float64) for w in m.W]
    m.b = [b.astype(np.float64) for b in m.b]
    y = (rng.random(20) > 0.5).astype(np.float64) if output == "sigmoid" else X

    # analytic gradient from the real backward pass (l2 = 0, dropout = 0)
    _, grads, _ = m._backward(X, y, rng)

    eps = 1e-6
    for layer in range(len(m.W)):
        for (i, j) in [(0, 0), (1, 0), (0, m.W[layer].shape[1] - 1)]:
            old = m.W[layer][i, j]
            m.W[layer][i, j] = old + eps
            lp = m._loss(m._forward(X, train=False)[0], y)
            m.W[layer][i, j] = old - eps
            lm = m._loss(m._forward(X, train=False)[0], y)
            m.W[layer][i, j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - grads[layer][i, j]) < 1e-5 * max(1.0, abs(num)), (output, layer, i, j, num, grads[layer][i, j])


def test_gradients():
    _numeric_grad_check("sigmoid")
    _numeric_grad_check("identity")


def test_dropout_scaling():
    rng = np.random.default_rng(0)
    X = np.abs(rng.normal(size=(20000, 3))).astype(np.float32)
    m = MLP(hidden=(50,), dropout=0.5, seed=0)
    m._init(3, 1)
    train_mean = np.mean([m._forward(X, train=True, rng=rng)[1][1].mean() for _ in range(5)])
    infer_mean = m._forward(X, train=False)[1][1].mean()
    assert abs(train_mean - infer_mean) / infer_mean < 0.02, (train_mean, infer_mean)


def test_learning():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] ** 2 > 1).astype(np.float32)
    m = MLP(hidden=(16,), dropout=0.1, lr=5e-3, epochs=30, batch_size=64, seed=0).fit(X, y)
    acc = np.mean((m.predict_proba(X)[:, 1] > 0.5) == y)
    assert acc > 0.9, acc

    normal = rng.normal(size=(4000, 6)).astype(np.float32) @ rng.normal(size=(6, 6)).astype(np.float32)
    ae = MLP(hidden=(8, 3, 8), output="identity", lr=3e-3, epochs=40, batch_size=64, seed=0).fit(normal[:3000])
    err_normal = ae.reconstruction_error(normal[3000:]).mean()
    err_anom = ae.reconstruction_error(normal[3000:] + 3.0).mean()
    assert err_anom > 2 * err_normal, (err_normal, err_anom)


if __name__ == "__main__":
    test_gradients(); test_dropout_scaling(); test_learning()
    print("all neural-network checks passed")
