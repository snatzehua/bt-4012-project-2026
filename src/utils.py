import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def strategy_for(best, model):
    """Imbalance strategy chosen for `model` in E1 (older runs stored one global string)."""
    st = best["strategy"]
    return st if isinstance(st, str) else st[model]
