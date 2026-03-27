# %%
"""
Ensemble size sweep — finds the optimal number of members.
Run this BEFORE final_pipeline.py to confirm N=5 is actually best,
or to pick a different number.

Strategy: rank all architectures by val_RMSE (best first), then
build ensembles of size 1, 2, 3, … N and measure test metrics each time.
This tells you the point of diminishing returns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.io import loadmat
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — same locked-in hyperparameters as final_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════
MAT_PATH = "data/compression_data_output.mat"
WINDOW   = 16

SHARED_PARAMS = dict(
    activation="tanh", solver="adam", alpha=5e-4,
    learning_rate_init=5e-4, max_iter=2500,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
)

# All architectures from Exp4, ordered best → worst by val_RMSE
# (paste your own df4 order here if different)
ALL_ARCHS = [
    (64, 32),        # rank 1 — best single model
    (256, 128, 64),  # rank 2
    (128, 64, 32),   # rank 3
    (64, 64),        # rank 4
    (256, 128),      # rank 5
    (128, 64),       # rank 6
    (128, 128),      # rank 7
    (128,),          # rank 8
    (64,),           # rank 9
    (32,),           # rank 10
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_window_data(signal_in, signal_out, window):
    a        = signal_in.ravel()
    y        = signal_out[:len(a)].reshape(-1, 1)
    a_padded = np.pad(a, (window - 1, 0), constant_values=0)
    X        = sliding_window_view(a_padded, window_shape=window, axis=0)
    return X[:len(y)], y[:len(X)]


def prepare_data(mat_path, window):
    sd      = loadmat(mat_path)
    X, y    = make_window_data(sd["train_input_real"], sd["train_output_real"], window)
    N       = len(X)
    X_dev,  y_dev  = X[:N//2], y[:N//2]
    X_test, y_test = X[N//2:], y[N//2:]
    sv      = int(0.8 * len(X_dev))
    X_train, y_train = X_dev[:sv], y_dev[:sv]
    X_val,   y_val   = X_dev[sv:], y_dev[sv:]
    X_mean  = X_train.mean(0, keepdims=True);  X_std = X_train.std(0, keepdims=True) + 1e-8
    y_mean  = y_train.mean(0, keepdims=True);  y_std = y_train.std(0, keepdims=True) + 1e-8
    def sx(a): return (a - X_mean) / X_std
    def sy(a): return (a - y_mean) / y_std
    return {"X_train": sx(X_train), "y_train": sy(y_train),
            "X_val":   sx(X_val),   "y_val":   sy(y_val),
            "X_test":  sx(X_test),  "y_test":  sy(y_test),
            "y_mean": y_mean, "y_std": y_std}


def unscale(y_s, d):
    return y_s * d["y_std"] + d["y_mean"]


def metrics(y_true, y_pred):
    yt, yp = np.ravel(y_true), np.ravel(y_pred)
    mse    = mean_squared_error(yt, yp)
    return {"RMSE": np.sqrt(mse), "MAE": mean_absolute_error(yt, yp),
            "R2": r2_score(yt, yp)}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — train all models once, collect val + test predictions
# ══════════════════════════════════════════════════════════════════════════════
print("Training all architectures …")
data = prepare_data(MAT_PATH, WINDOW)
yt   = np.ravel(unscale(data["y_test"], data))
yv   = np.ravel(unscale(data["y_val"],  data))

val_preds  = []
test_preds = []

for i, arch in enumerate(ALL_ARCHS):
    print(f"  [{i+1}/{len(ALL_ARCHS)}] arch={arch}")
    mlp = MLPRegressor(**SHARED_PARAMS, hidden_layer_sizes=arch, random_state=42)
    mlp.fit(data["X_train"], np.ravel(data["y_train"]))

    vp = np.ravel(unscale(mlp.predict(data["X_val"]).reshape(-1,1),  data))
    tp = np.ravel(unscale(mlp.predict(data["X_test"]).reshape(-1,1), data))

    val_m  = metrics(yv, vp)
    test_m = metrics(yt, tp)
    val_preds.append(vp)
    test_preds.append(tp)
    print(f"         val_RMSE={val_m['RMSE']:,.0f}  "
          f"test_RMSE={test_m['RMSE']:,.0f}  R²={test_m['R2']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — sweep ensemble size 1 … N
#          members are added in rank order (best val_RMSE first)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("ENSEMBLE SIZE SWEEP")
print("=" * 55)
print(f"  {'N':>4}  {'val_RMSE':>10}  {'test_RMSE':>10}  {'test_R2':>10}  {'test_MAE':>10}")
print("  " + "-" * 50)

sweep_results = []
for n in range(1, len(ALL_ARCHS) + 1):
    ens_val  = np.mean(val_preds[:n],  axis=0)
    ens_test = np.mean(test_preds[:n], axis=0)
    vm = metrics(yv, ens_val)
    tm = metrics(yt, ens_test)
    sweep_results.append({"N": n, "arch_added": str(ALL_ARCHS[n-1]),
                           "val_RMSE":  vm["RMSE"], "test_RMSE": tm["RMSE"],
                           "test_R2":   tm["R2"],   "test_MAE":  tm["MAE"]})
    print(f"  {n:>4}  {vm['RMSE']:>10,.1f}  {tm['RMSE']:>10,.1f}  "
          f"{tm['R2']:>10.6f}  {tm['MAE']:>10,.1f}   + {ALL_ARCHS[n-1]}")

df_sweep = pd.DataFrame(sweep_results)
best_n   = int(df_sweep.loc[df_sweep["val_RMSE"].idxmin(), "N"])

print(f"\n  → Best ensemble size by val_RMSE: N = {best_n}")
print(f"    val_RMSE  = {df_sweep.loc[best_n-1,'val_RMSE']:,.1f}")
print(f"    test_RMSE = {df_sweep.loc[best_n-1,'test_RMSE']:,.1f}")
print(f"    test_R2   = {df_sweep.loc[best_n-1,'test_R2']:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — also try: sort by DIVERSITY instead of rank
#           (greedy: each new member maximises disagreement with current ensemble)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("GREEDY DIVERSITY ENSEMBLE  (most-disagreeing members first)")
print("=" * 55)

remaining   = list(range(len(ALL_ARCHS)))
chosen      = [0]   # always start with the best single model
remaining.remove(0)

div_results = []
ens_pred    = test_preds[0].copy()
vm0 = metrics(yv, val_preds[0]);  tm0 = metrics(yt, test_preds[0])
div_results.append({"N": 1, "val_RMSE": vm0["RMSE"], "test_RMSE": tm0["RMSE"],
                    "test_R2": tm0["R2"]})
print(f"  N=1  added {ALL_ARCHS[0]}  test_RMSE={tm0['RMSE']:,.1f}")

while remaining:
    # pick next member whose predictions disagree most with current ensemble mean
    current_ens = np.mean([test_preds[i] for i in chosen], axis=0)
    disagreement = {i: np.std(test_preds[i] - current_ens) for i in remaining}
    next_i = max(disagreement, key=disagreement.get)
    chosen.append(next_i)
    remaining.remove(next_i)

    ens_v = np.mean([val_preds[i]  for i in chosen], axis=0)
    ens_t = np.mean([test_preds[i] for i in chosen], axis=0)
    vm = metrics(yv, ens_v);  tm = metrics(yt, ens_t)
    div_results.append({"N": len(chosen), "val_RMSE": vm["RMSE"],
                        "test_RMSE": tm["RMSE"], "test_R2": tm["R2"]})
    print(f"  N={len(chosen)}  added {ALL_ARCHS[next_i]}  "
          f"val_RMSE={vm['RMSE']:,.1f}  test_RMSE={tm['RMSE']:,.1f}")

df_div   = pd.DataFrame(div_results)
best_n_d = int(df_div.loc[df_div["val_RMSE"].idxmin(), "N"])
print(f"\n  → Best diversity ensemble size: N = {best_n_d}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT — ensemble size vs RMSE for both strategies
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8,
                     "figure.dpi": 150, "xtick.direction": "in",
                     "ytick.direction": "in"})

fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
fig.suptitle("Fig. 7 — Ensemble size sweep: diminishing returns analysis",
             fontsize=9, fontweight="bold")

for ax, df, title, best in [
    (axes[0], df_sweep, "(a) Rank-order ensemble", best_n),
    (axes[1], df_div,   "(b) Greedy diversity ensemble", best_n_d),
]:
    ax.plot(df["N"], df["val_RMSE"],  color="#2e86ab", lw=1.3,
            marker="o", ms=4, label="Val RMSE")
    ax.plot(df["N"], df["test_RMSE"], color="#e84545", lw=1.3,
            marker="s", ms=4, ls="--", label="Test RMSE")
    ax.axvline(best, color="k", lw=0.9, ls=":", label=f"Best N={best}")
    ax.set_xlabel("Number of ensemble members")
    ax.set_ylabel("RMSE (a.u.)")
    ax.set_title(title)
    ax.set_xticks(df["N"])
    ax.legend(framealpha=0.7)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

fig.tight_layout()
fig.savefig("data/fig7_ensemble_size_sweep.png", dpi=200, bbox_inches="tight")
plt.show()

print(f"\n{'='*55}")
print(f"RECOMMENDATION")
print(f"{'='*55}")
print(f"  Rank-order best N    : {best_n}  "
      f"(test_RMSE={df_sweep.loc[best_n-1,'test_RMSE']:,.1f})")
print(f"  Diversity best N     : {best_n_d}  "
      f"(test_RMSE={df_div.loc[best_n_d-1,'test_RMSE']:,.1f})")
print(f"\n  Update TOP5_ARCHS in final_pipeline.py with the best N")
print(f"  from whichever strategy gives lower val_RMSE.")
