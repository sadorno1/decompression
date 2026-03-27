# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy.io import loadmat
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ══════════════════════════════════════════════════════════════════════════════
# JOURNAL STYLE — applied once, affects all figures
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif"],
    "font.size":           9,
    "axes.labelsize":      9,
    "axes.titlesize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     8,
    "figure.dpi":          150,
    "axes.linewidth":      0.8,
    "grid.linewidth":      0.4,
    "lines.linewidth":     1.0,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})

PANEL_W  = 3.3   # one-column figure width (inches)
PANEL_2C = 6.8   # two-column figure width (inches)
COLOR_ACT = "#1a1a2e"
COLOR_PRD = "#e84545"
COLOR_SC  = "#2e86ab"
COLOR_RES = "#f18f01"

# ══════════════════════════════════════════════════════════════════════════════
# CORE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_window_data(signal_in, signal_out, window):
    a = signal_in.ravel()
    y = signal_out[:len(a)].reshape(-1, 1)
    a_padded = np.pad(a, (window - 1, 0), constant_values=0)
    X = sliding_window_view(a_padded, window_shape=window, axis=0)
    X = X[:len(y)]
    y = y[:len(X)]
    return X, y


def prepare_data(mat_path, window, extra_features=False):
    signal_data = loadmat(mat_path)
    x_raw = signal_data["train_input_real"]
    y_raw = signal_data["train_output_real"]
    X_raw, y = make_window_data(x_raw, y_raw, window=window)

    if extra_features:
        win_mean   = X_raw.mean(axis=1, keepdims=True)
        win_std    = X_raw.std(axis=1,  keepdims=True) + 1e-8
        win_min    = X_raw.min(axis=1,  keepdims=True)
        win_max    = X_raw.max(axis=1,  keepdims=True)
        win_rng    = win_max - win_min
        delta_mean = np.diff(X_raw, axis=1).mean(axis=1, keepdims=True)
        X = np.hstack([X_raw, win_mean, win_std, win_min, win_max, win_rng, delta_mean])
    else:
        X = X_raw

    N = len(X)
    X_dev,  y_dev  = X[:N//2],  y[:N//2]
    X_test, y_test = X[N//2:],  y[N//2:]

    split_val = int(0.8 * len(X_dev))
    X_train, y_train = X_dev[:split_val], y_dev[:split_val]
    X_val,   y_val   = X_dev[split_val:], y_dev[split_val:]

    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std  = X_train.std(axis=0,  keepdims=True) + 1e-8
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std  = y_train.std(axis=0,  keepdims=True) + 1e-8

    return {
        "X_train": (X_train - X_mean) / X_std,
        "y_train": (y_train - y_mean) / y_std,
        "X_val":   (X_val   - X_mean) / X_std,
        "y_val":   (y_val   - y_mean) / y_std,
        "X_test":  (X_test  - X_mean) / X_std,
        "y_test":  (y_test  - y_mean) / y_std,
        "X_mean": X_mean, "X_std": X_std,
        "y_mean": y_mean, "y_std": y_std,
    }


def unscale_y(y_s, d):
    return y_s * d["y_std"] + d["y_mean"]


def compute_metrics(y_true, y_pred):
    yt, yp = np.ravel(y_true), np.ravel(y_pred)
    mse = mean_squared_error(yt, yp)
    return {"MSE": mse, "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(yt, yp),
            "R2":  r2_score(yt, yp)}


def train_mlp(data, params, random_state=42):
    mlp = MLPRegressor(**params, random_state=random_state)
    mlp.fit(data["X_train"], np.ravel(data["y_train"]))

    def pred(X): return unscale_y(mlp.predict(X).reshape(-1, 1), data)

    return mlp, {
        "train":    compute_metrics(unscale_y(data["y_train"], data), pred(data["X_train"])),
        "val":      compute_metrics(unscale_y(data["y_val"],   data), pred(data["X_val"])),
        "test":     compute_metrics(unscale_y(data["y_test"],  data), pred(data["X_test"])),
        "y_test":   unscale_y(data["y_test"], data),
        "yhat_test": pred(data["X_test"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Fine window search
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("EXPERIMENT 1: Fine window search (8–24, steps of 2)")
print("=" * 60)

best_params = dict(
    hidden_layer_sizes=(64, 32), activation="tanh", solver="adam",
    alpha=1e-4, learning_rate_init=5e-4, max_iter=2500,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
)

exp1_results = []
for w in range(8, 26, 2):
    data = prepare_data("data/compression_data_output.mat", window=w)
    _, res = train_mlp(data, best_params)
    exp1_results.append({"window": w, "val_RMSE": res["val"]["RMSE"],
                          "test_RMSE": res["test"]["RMSE"], "test_R2": res["test"]["R2"]})
    print(f"  w={w:2d}  val_RMSE={res['val']['RMSE']:,.0f}  "
          f"test_RMSE={res['test']['RMSE']:,.0f}  R²={res['test']['R2']:.4f}")

df1 = pd.DataFrame(exp1_results).sort_values("val_RMSE")
best_window = int(df1.iloc[0]["window"])
print(f"\n  → Best window: {best_window}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Extra features
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 2: Extra features (mean, std, min, max, delta)")
print("=" * 60)

exp2_results = []
for ef in [False, True]:
    data = prepare_data("data/compression_data_output.mat", window=best_window, extra_features=ef)
    _, res = train_mlp(data, best_params)
    label = f"w={best_window} {'+ stats' if ef else 'raw   '}"
    exp2_results.append({"label": label, "val_RMSE": res["val"]["RMSE"],
                          "test_RMSE": res["test"]["RMSE"], "test_R2": res["test"]["R2"]})
    print(f"  {label}  val_RMSE={res['val']['RMSE']:,.0f}  "
          f"test_RMSE={res['test']['RMSE']:,.0f}  R²={res['test']['R2']:.4f}")

df2 = pd.DataFrame(exp2_results).sort_values("val_RMSE")
use_extra = "+ stats" in df2.iloc[0]["label"]
print(f"\n  → Extra features help: {use_extra}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Alpha sweep
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 3: Alpha (regularisation) sweep")
print("=" * 60)

data_main = prepare_data("data/compression_data_output.mat",
                          window=best_window, extra_features=use_extra)

exp3_results = []
for alpha in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
    _, res = train_mlp(data_main, {**best_params, "alpha": alpha})
    exp3_results.append({"alpha": alpha, "val_RMSE": res["val"]["RMSE"],
                          "test_RMSE": res["test"]["RMSE"], "test_R2": res["test"]["R2"]})
    print(f"  alpha={alpha:.0e}  val_RMSE={res['val']['RMSE']:,.0f}  "
          f"test_RMSE={res['test']['RMSE']:,.0f}  R²={res['test']['R2']:.4f}")

df3 = pd.DataFrame(exp3_results).sort_values("val_RMSE")
best_alpha = float(df3.iloc[0]["alpha"])
print(f"\n  → Best alpha: {best_alpha:.0e}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — Architecture search
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 4: Architecture search")
print("=" * 60)

architectures = [(32,), (64,), (128,), (64,32), (128,64),
                 (128,64,32), (64,64), (128,128), (256,128), (256,128,64)]

exp4_results = []
for arch in architectures:
    _, res = train_mlp(data_main, {**best_params, "alpha": best_alpha,
                                    "hidden_layer_sizes": arch})
    exp4_results.append({"arch": str(arch), "val_RMSE": res["val"]["RMSE"],
                          "test_RMSE": res["test"]["RMSE"], "test_R2": res["test"]["R2"]})
    print(f"  arch={str(arch):20s}  val_RMSE={res['val']['RMSE']:,.0f}  "
          f"test_RMSE={res['test']['RMSE']:,.0f}  R²={res['test']['R2']:.4f}")

df4 = pd.DataFrame(exp4_results).sort_values("val_RMSE")
best_arch = eval(df4.iloc[0]["arch"])
print(f"\n  → Best architecture: {best_arch}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5 — Architecture ensemble (top 5)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 5: Ensemble of top-5 architectures")
print("=" * 60)

ens_preds = []
best_model = None  # keep one model for the loss curve plot

for arch in [eval(r) for r in df4.head(5)["arch"]]:
    mlp, res = train_mlp(data_main, {**best_params, "alpha": best_alpha,
                                      "hidden_layer_sizes": arch})
    ens_preds.append(np.ravel(res["yhat_test"]))
    if best_model is None:
        best_model = mlp   # first = best single model

yt_ens = np.ravel(unscale_y(data_main["y_test"], data_main))
ens_pred = np.mean(ens_preds, axis=0)
ens_metrics = compute_metrics(yt_ens, ens_pred)
print(f"  Ensemble  RMSE={ens_metrics['RMSE']:,.0f}  "
      f"MAE={ens_metrics['MAE']:,.0f}  R²={ens_metrics['R2']:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6 — XGBoost
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 6: XGBoost")
print("=" * 60)

try:
    from xgboost import XGBRegressor
    for cfg in [{"n_estimators": 500,  "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
                {"n_estimators": 1000, "max_depth": 6, "learning_rate": 0.02, "subsample": 0.8},
                {"n_estimators": 1000, "max_depth": 8, "learning_rate": 0.01, "subsample": 0.9}]:
        xgb_m = XGBRegressor(**cfg, n_jobs=-1, random_state=42, verbosity=0,
                              early_stopping_rounds=30, eval_metric="rmse")
        xgb_m.fit(data_main["X_train"], np.ravel(data_main["y_train"]),
                  eval_set=[(data_main["X_val"], np.ravel(data_main["y_val"]))], verbose=False)
        yhat_te = unscale_y(xgb_m.predict(data_main["X_test"]).reshape(-1,1), data_main)
        yhat_va = unscale_y(xgb_m.predict(data_main["X_val"]).reshape(-1,1),  data_main)
        vm = compute_metrics(unscale_y(data_main["y_test"], data_main), yhat_te)
        tm = compute_metrics(unscale_y(data_main["y_val"],  data_main), yhat_va)
        print(f"  depth={cfg['max_depth']} n={cfg['n_estimators']} lr={cfg['learning_rate']}  "
              f"val_RMSE={tm['RMSE']:,.0f}  test_RMSE={vm['RMSE']:,.0f}  R²={vm['R2']:.4f}")
except ImportError:
    print("  XGBoost not installed — run: pip install xgboost")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7 — Multi-seed stability + seed ensemble
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 7: Multi-seed stability (5 seeds)")
print("=" * 60)

final_params = {**best_params, "alpha": best_alpha, "hidden_layer_sizes": best_arch}
seed_preds, seed_rows = [], []

for seed in [0, 7, 13, 42, 99]:
    mlp = MLPRegressor(**final_params, random_state=seed)
    mlp.fit(data_main["X_train"], np.ravel(data_main["y_train"]))
    yhat = unscale_y(mlp.predict(data_main["X_test"]).reshape(-1,1), data_main)
    y_te = unscale_y(data_main["y_test"], data_main)
    m = compute_metrics(y_te, yhat)
    seed_rows.append({"seed": seed, **m})
    seed_preds.append(np.ravel(yhat))
    print(f"  seed={seed:2d}  RMSE={m['RMSE']:,.0f}  R²={m['R2']:.6f}")

df7 = pd.DataFrame(seed_rows)
print(f"\n  RMSE  mean={df7['RMSE'].mean():,.0f}  std={df7['RMSE'].std():,.0f}")
print(f"  R²    mean={df7['R2'].mean():.6f}  std={df7['R2'].std():.6f}")

seed_ens  = np.mean(seed_preds, axis=0)
yt_s      = np.ravel(unscale_y(data_main["y_test"], data_main))
sm        = compute_metrics(yt_s, seed_ens)
print(f"\n  Seed-ensemble  RMSE={sm['RMSE']:,.0f}  R²={sm['R2']:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

summary = pd.DataFrame([
    {"Experiment": "Original best (w=16, (64,32) tanh)", "test_RMSE": 844.0,                      "test_R2": 0.995854},
    {"Experiment": f"Exp1: Best window={best_window}",   "test_RMSE": df1.iloc[0]["test_RMSE"],   "test_R2": df1.iloc[0]["test_R2"]},
    {"Experiment": f"Exp3: Best alpha={best_alpha:.0e}", "test_RMSE": df3.iloc[0]["test_RMSE"],   "test_R2": df3.iloc[0]["test_R2"]},
    {"Experiment": f"Exp4: Best arch={best_arch}",       "test_RMSE": df4.iloc[0]["test_RMSE"],   "test_R2": df4.iloc[0]["test_R2"]},
    {"Experiment": "Exp5: Architecture ensemble",        "test_RMSE": ens_metrics["RMSE"],         "test_R2": ens_metrics["R2"]},
    {"Experiment": "Exp7: Seed ensemble",                "test_RMSE": sm["RMSE"],                  "test_R2": sm["R2"]},
])
print(summary.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES  — use architecture ensemble predictions (best result)
# ══════════════════════════════════════════════════════════════════════════════
yt  = yt_ens
yp  = ens_pred
res = yt - yp


# ── Fig 1: Predicted vs Actual ────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(PANEL_W, PANEL_W))
ax.scatter(yt, yp, s=1.5, alpha=0.25, color=COLOR_SC, linewidths=0, rasterized=True)
lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label=r"Ideal ($y = \hat{y}$)")
ax.set_xlabel("Actual output amplitude (a.u.)")
ax.set_ylabel("Predicted output amplitude (a.u.)")
ax.set_title("Fig. 1 — Predicted vs. actual output\n(ensemble MLP, hold-out test set)")
ax.legend(loc="upper left", framealpha=0.7)
ax.text(0.97, 0.05, f"$R^2 = {ens_metrics['R2']:.4f}$\nRMSE $= {ens_metrics['RMSE']:,.0f}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(fc="white", ec="0.7", pad=3, lw=0.6))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
fig1.tight_layout()
fig1.savefig("data/fig1_predicted_vs_actual.png", dpi=200, bbox_inches="tight")


# ── Fig 2: Residual vs Amplitude ──────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(PANEL_W, PANEL_W))
ax.scatter(np.abs(yt), np.abs(res), s=1.5, alpha=0.2,
           color=COLOR_RES, linewidths=0, rasterized=True)
sort_idx = np.argsort(np.abs(yt))
x_s = np.abs(yt)[sort_idx];  r_s = np.abs(res)[sort_idx]
win = max(1, len(x_s) // 200)
roll = np.convolve(r_s, np.ones(win)/win, mode="valid")
ax.plot(x_s[win//2: win//2+len(roll)], roll, color="k", lw=1.4, label="Moving avg.")
ax.set_xlabel(r"Signal amplitude $|y|$ (a.u.)")
ax.set_ylabel(r"Absolute residual $|y - \hat{y}|$ (a.u.)")
ax.set_title("Fig. 2 — Residual magnitude vs. signal amplitude\n(heteroscedasticity check)")
ax.legend(loc="upper left", framealpha=0.7)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
fig2.tight_layout()
fig2.savefig("data/fig2_residual_vs_amplitude.png", dpi=200, bbox_inches="tight")


# ── Fig 3: Error distribution ─────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(PANEL_W, PANEL_W * 0.85))
ax.hist(res, bins=150, color=COLOR_SC, edgecolor="none", alpha=0.85, density=True)
ax.axvline(0,              color="k",       lw=1.2, ls="--", label="Zero error")
ax.axvline(res.mean(),     color=COLOR_PRD, lw=1.2, ls="-",  label=f"Mean = {res.mean():+.0f}")
ax.axvline(np.median(res), color=COLOR_RES, lw=1.2, ls="-.", label=f"Median = {np.median(res):+.0f}")
ax.set_xlabel(r"Prediction error $y - \hat{y}$ (a.u.)")
ax.set_ylabel("Probability density")
ax.set_title(f"Fig. 3 — Prediction error distribution\n(hold-out test set, $n$ = {len(res):,})")
ax.legend(framealpha=0.7)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
fig3.tight_layout()
fig3.savefig("data/fig3_error_distribution.png", dpi=200, bbox_inches="tight")


# ── Fig 4: 4-panel waveform zoom (50 pts, one per amplitude quartile) ─────────
WIN = 50
amp_abs = np.abs(yt)
q25, q50, q75 = np.percentile(amp_abs, [25, 50, 75])
q_bounds = [(0, q25), (q25, q50), (q50, q75), (q75, amp_abs.max()*1.01)]
qlabels  = ["Low amplitude", "Medium-low amplitude",
            "Medium-high amplitude", "High amplitude"]

def find_segment(amp, lo, hi, win=50, n=500):
    step = max(1, len(amp) // n)
    candidates = [i for i in range(0, len(amp)-win, step)
                  if lo <= amp[i:i+win].mean() < hi]
    if not candidates:
        all_m = [amp[i:i+win].mean() for i in range(0, len(amp)-win, step)]
        candidates = [np.argmin(np.abs(np.array(all_m) - (lo+hi)/2)) * step]
    return min(candidates, key=lambda i: abs(amp[i:i+win].mean() - (lo+hi)/2))

starts = [find_segment(amp_abs, lo, hi) for lo, hi in q_bounds]

fig4 = plt.figure(figsize=(PANEL_2C, PANEL_2C * 0.7))
gs   = gridspec.GridSpec(2, 2, figure=fig4, hspace=0.48, wspace=0.38)

for idx, (start, qlabel) in enumerate(zip(starts, qlabels)):
    ax = fig4.add_subplot(gs[idx // 2, idx % 2])
    t  = np.arange(WIN)
    ax.plot(t, yt[start:start+WIN], color=COLOR_ACT, lw=1.2, label="Actual",    zorder=3)
    ax.plot(t, yp[start:start+WIN], color=COLOR_PRD, lw=1.2, ls="--",
            label="Predicted", zorder=4)
    ax.fill_between(t, yt[start:start+WIN], yp[start:start+WIN],
                    alpha=0.15, color=COLOR_PRD, zorder=2)
    local_rmse = np.sqrt(np.mean((yt[start:start+WIN] - yp[start:start+WIN])**2))
    letter = ["(a)","(b)","(c)","(d)"][idx]
    ax.set_title(f"{letter} {qlabel}\nsamples {start}–{start+WIN-1},"
                 f" local RMSE = {local_rmse:,.0f}", pad=4)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if idx == 0:
        ax.legend(loc="upper right", framealpha=0.7, handlelength=1.6)

fig4.suptitle("Fig. 4 — Waveform reconstruction across amplitude regimes\n"
              "(50-sample windows; shaded area = prediction error)",
              fontsize=9, fontweight="bold", y=1.01)
fig4.savefig("data/fig4_waveform_zoom.png", dpi=200, bbox_inches="tight")


# ── Fig 5: Training loss curve ────────────────────────────────────────────────
fig5, ax = plt.subplots(figsize=(PANEL_W, PANEL_W * 0.75))
ax.plot(best_model.loss_curve_, color=COLOR_ACT, lw=1.1)
n_ep = len(best_model.loss_curve_)
ax.axvline(n_ep, color=COLOR_PRD, lw=0.9, ls="--", label=f"Stop at epoch {n_ep}")
ax.set_xlabel("Training epoch")
ax.set_ylabel("Training loss (MSE, scaled units)")
ax.set_title("Fig. 5 — Training loss curve\n(best single MLP; early stopping)")
ax.legend(framealpha=0.7)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
fig5.tight_layout()
fig5.savefig("data/fig5_loss_curve.png", dpi=200, bbox_inches="tight")

plt.show()
print("\nDone. Figures saved to data/")
print("  fig1_predicted_vs_actual.png")
print("  fig2_residual_vs_amplitude.png")
print("  fig3_error_distribution.png")
print("  fig4_waveform_zoom.png")
print("  fig5_loss_curve.png")
