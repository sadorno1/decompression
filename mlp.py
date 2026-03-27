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
# FINAL MODEL CONFIG  ← change anything here if needed
# ══════════════════════════════════════════════════════════════════════════════
MAT_PATH   = "data/compression_data_output.mat"
WINDOW     = 16
TOP5_ARCHS = [(64,32), (256,128,64), (128,64,32), (64,64), (256,128)]  # architecture ensemble

SHARED_PARAMS = dict(
    activation="tanh", solver="adam", alpha=5e-4,
    learning_rate_init=5e-4, max_iter=2500,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
)

# ══════════════════════════════════════════════════════════════════════════════
# JOURNAL STYLE
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif"],
    "font.size":           9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize":     8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi":          150, "axes.linewidth": 0.8, "lines.linewidth": 1.0,
    "xtick.direction":     "in", "ytick.direction": "in",
    "xtick.minor.visible": True, "ytick.minor.visible": True,
})

PANEL_W  = 3.3
PANEL_2C = 6.8
COLOR_ACT = "#1a1a2e"
COLOR_PRD = "#e84545"
COLOR_SC  = "#2e86ab"
COLOR_RES = "#f18f01"

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
    sd    = loadmat(mat_path)
    X, y  = make_window_data(sd["train_input_real"], sd["train_output_real"], window)

    N            = len(X)
    X_dev, y_dev = X[:N//2], y[:N//2]
    X_test, y_test = X[N//2:], y[N//2:]

    sv           = int(0.8 * len(X_dev))
    X_train, y_train = X_dev[:sv], y_dev[:sv]
    X_val,   y_val   = X_dev[sv:], y_dev[sv:]

    X_mean = X_train.mean(0, keepdims=True);  X_std = X_train.std(0, keepdims=True) + 1e-8
    y_mean = y_train.mean(0, keepdims=True);  y_std = y_train.std(0, keepdims=True) + 1e-8

    def sx(a): return (a - X_mean) / X_std
    def sy(a): return (a - y_mean) / y_std

    return {"X_train": sx(X_train), "y_train": sy(y_train),
            "X_val":   sx(X_val),   "y_val":   sy(y_val),
            "X_test":  sx(X_test),  "y_test":  sy(y_test),
            "y_mean": y_mean,       "y_std":   y_std}


def unscale(y_s, d):
    return y_s * d["y_std"] + d["y_mean"]


def metrics(y_true, y_pred):
    yt, yp = np.ravel(y_true), np.ravel(y_pred)
    mse    = mean_squared_error(yt, yp)
    return {"MSE": mse, "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(yt, yp), "R2": r2_score(yt, yp)}

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("Training architecture ensemble …")
data = prepare_data(MAT_PATH, WINDOW)

ens_preds   = []
loss_curves = []

for i, arch in enumerate(TOP5_ARCHS):
    print(f"  [{i+1}/{len(TOP5_ARCHS)}] arch={arch}")
    mlp = MLPRegressor(**SHARED_PARAMS, hidden_layer_sizes=arch, random_state=42)
    mlp.fit(data["X_train"], np.ravel(data["y_train"]))
    yhat_s = mlp.predict(data["X_test"]).reshape(-1, 1)
    ens_preds.append(np.ravel(unscale(yhat_s, data)))
    loss_curves.append(mlp.loss_curve_)

yt       = np.ravel(unscale(data["y_test"], data))
yp       = np.mean(ens_preds, axis=0)
res      = yt - yp
m        = metrics(yt, yp)

# individual model metrics too
ind_metrics = []
for i, (arch, pred) in enumerate(zip(TOP5_ARCHS, ens_preds)):
    im = metrics(yt, pred)
    ind_metrics.append({"arch": str(arch), **im})

# ══════════════════════════════════════════════════════════════════════════════
# METRICS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("INDIVIDUAL MODEL METRICS")
print("=" * 55)
df_ind = pd.DataFrame(ind_metrics)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
print(df_ind.to_string(index=False))

print("\n" + "=" * 55)
print("ENSEMBLE METRICS  (architecture ensemble, test set)")
print("=" * 55)
print(f"  MSE  : {m['MSE']:>15,.2f}")
print(f"  RMSE : {m['RMSE']:>15,.2f}")
print(f"  MAE  : {m['MAE']:>15,.2f}")
print(f"  R²   : {m['R2']:>15.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# RELATIVE ERROR
# ══════════════════════════════════════════════════════════════════════════════
amp_threshold = 0.01 * np.abs(yt).max()
valid         = np.abs(yt) > amp_threshold
rel_err_pct   = np.abs(res[valid]) / np.abs(yt[valid]) * 100

mape  = rel_err_pct.mean()
mdape = np.median(rel_err_pct)
p95   = np.percentile(rel_err_pct, 95)
p99   = np.percentile(rel_err_pct, 99)

print("\n" + "=" * 55)
print("RELATIVE ERROR  (samples with |y| > 1% of peak)")
print("=" * 55)
print(f"  Valid samples       : {valid.sum():,} / {len(yt):,}  ({100*valid.mean():.1f}%)")
print(f"  MAPE                : {mape:.2f}%")
print(f"  Median APE          : {mdape:.2f}%")
print(f"  95th percentile APE : {p95:.2f}%")
print(f"  99th percentile APE : {p99:.2f}%")

amp_v    = np.abs(yt[valid])
re_v     = rel_err_pct
q25v, q50v, q75v = np.percentile(amp_v, [25, 50, 75])
qbins    = [(0, q25v, "Low (Q1)"), (q25v, q50v, "Med-low (Q2)"),
            (q50v, q75v, "Med-high (Q3)"), (q75v, amp_v.max()*1.01, "High (Q4)")]

print(f"\n  {'Bin':<18} {'N':>7}  {'MAPE':>7}  {'Median':>7}  {'95th':>7}")
print("  " + "-" * 52)
bin_stats = []
for lo, hi, label in qbins:
    mask = (amp_v >= lo) & (amp_v < hi)
    rp   = re_v[mask]
    row  = {"bin": label, "n": int(mask.sum()), "mape": rp.mean(),
            "mdape": np.median(rp), "p95": np.percentile(rp, 95)}
    bin_stats.append(row)
    print(f"  {label:<18} {mask.sum():>7,}  {rp.mean():>6.2f}%  "
          f"{np.median(rp):>6.2f}%  {np.percentile(rp, 95):>6.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── Fig 1: Predicted vs Actual ────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(PANEL_W, PANEL_W))
ax.scatter(yt, yp, s=1.5, alpha=0.25, color=COLOR_SC, linewidths=0, rasterized=True)
lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label=r"Ideal ($y=\hat{y}$)")
ax.set_xlabel("Actual output amplitude (a.u.)")
ax.set_ylabel("Predicted output amplitude (a.u.)")
ax.set_title("Fig. 1 — Predicted vs. actual output\n(ensemble MLP, hold-out test set)")
ax.legend(loc="upper left", framealpha=0.7)
ax.text(0.97, 0.05, f"$R^2 = {m['R2']:.4f}$\nRMSE $= {m['RMSE']:,.0f}$",
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
si  = np.argsort(np.abs(yt))
xs  = np.abs(yt)[si];  rs = np.abs(res)[si]
win = max(1, len(xs) // 200)
ax.plot(xs[win//2: win//2 + len(np.convolve(rs, np.ones(win)/win, "valid"))],
        np.convolve(rs, np.ones(win)/win, "valid"), color="k", lw=1.4, label="Moving avg.")
ax.set_xlabel(r"Signal amplitude $|y|$ (a.u.)")
ax.set_ylabel(r"Absolute residual $|y-\hat{y}|$ (a.u.)")
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
ax.set_xlabel(r"Prediction error $y-\hat{y}$ (a.u.)")
ax.set_ylabel("Probability density")
ax.set_title(f"Fig. 3 — Prediction error distribution\n(hold-out test set, $n$ = {len(res):,})")
ax.legend(framealpha=0.7)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
fig3.tight_layout()
fig3.savefig("data/fig3_error_distribution.png", dpi=200, bbox_inches="tight")

# ── Fig 4: 4-panel waveform zoom ──────────────────────────────────────────────
WIN = 50
amp_abs = np.abs(yt)
q25, q50, q75 = np.percentile(amp_abs, [25, 50, 75])

def find_segment(amp, lo, hi, win=50, n=500):
    step = max(1, len(amp) // n)
    cands = [i for i in range(0, len(amp)-win, step)
             if lo <= amp[i:i+win].mean() < hi]
    if not cands:
        all_m = [amp[i:i+win].mean() for i in range(0, len(amp)-win, step)]
        cands = [np.argmin(np.abs(np.array(all_m) - (lo+hi)/2)) * step]
    return min(cands, key=lambda i: abs(amp[i:i+win].mean() - (lo+hi)/2))

starts  = [find_segment(amp_abs, lo, hi) for lo, hi in
           [(0,q25),(q25,q50),(q50,q75),(q75,amp_abs.max()*1.01)]]
qlabels = ["Low amplitude","Medium-low amplitude","Medium-high amplitude","High amplitude"]

fig4 = plt.figure(figsize=(PANEL_2C, PANEL_2C * 0.7))
gs   = gridspec.GridSpec(2, 2, figure=fig4, hspace=0.48, wspace=0.38)
for idx, (start, qlabel) in enumerate(zip(starts, qlabels)):
    ax = fig4.add_subplot(gs[idx//2, idx%2])
    t  = np.arange(WIN)
    ax.plot(t, yt[start:start+WIN], color=COLOR_ACT, lw=1.2, label="Actual",    zorder=3)
    ax.plot(t, yp[start:start+WIN], color=COLOR_PRD, lw=1.2, ls="--", label="Predicted", zorder=4)
    ax.fill_between(t, yt[start:start+WIN], yp[start:start+WIN],
                    alpha=0.15, color=COLOR_PRD, zorder=2)
    local_rmse = np.sqrt(np.mean((yt[start:start+WIN] - yp[start:start+WIN])**2))
    letter = ["(a)","(b)","(c)","(d)"][idx]
    ax.set_title(f"{letter} {qlabel}\nsamples {start}–{start+WIN-1}, "
                 f"local RMSE = {local_rmse:,.0f}", pad=4)
    ax.set_xlabel("Sample index");  ax.set_ylabel("Amplitude (a.u.)")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if idx == 0:
        ax.legend(loc="upper right", framealpha=0.7, handlelength=1.6)

fig4.suptitle("Fig. 4 — Waveform reconstruction across amplitude regimes\n"
              "(50-sample windows; shaded area = prediction error)",
              fontsize=9, fontweight="bold", y=1.01)
fig4.savefig("data/fig4_waveform_zoom.png", dpi=200, bbox_inches="tight")

# ── Fig 5: Loss curves (all 5 ensemble members) ───────────────────────────────
fig5, ax = plt.subplots(figsize=(PANEL_W, PANEL_W * 0.75))
colors5 = ["#1a1a2e","#2e86ab","#e84545","#f18f01","#4caf50"]
for i, (curve, arch) in enumerate(zip(loss_curves, TOP5_ARCHS)):
    ax.plot(curve, color=colors5[i], lw=0.9, alpha=0.85, label=str(arch))
ax.set_xlabel("Training epoch")
ax.set_ylabel("Training loss (MSE, scaled units)")
ax.set_title("Fig. 5 — Training loss curves\n(all 5 ensemble members)")
ax.legend(fontsize=7, framealpha=0.7)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
fig5.tight_layout()
fig5.savefig("data/fig5_loss_curves.png", dpi=200, bbox_inches="tight")

# ── Fig 6: Relative error (3-panel) ──────────────────────────────────────────
fig6, axes = plt.subplots(1, 3, figsize=(PANEL_2C, PANEL_2C * 0.42))
fig6.suptitle("Fig. 6 — Relative prediction error (% of signal amplitude)",
              fontsize=9, fontweight="bold")

# (a) scatter
ax = axes[0]
ax.scatter(amp_v, re_v, s=1.2, alpha=0.15, color=COLOR_RES, linewidths=0, rasterized=True)
si2   = np.argsort(amp_v)
xs2   = amp_v[si2];  rs2 = re_v[si2]
win2  = max(1, len(xs2) // 200)
roll2 = np.convolve(rs2, np.ones(win2)/win2, "valid")
ax.plot(xs2[win2//2: win2//2+len(roll2)], roll2, color="k", lw=1.4, label="Moving avg.")
ax.set_xlabel(r"Signal amplitude $|y|$ (a.u.)")
ax.set_ylabel("Relative error (% of amplitude)")
ax.set_title("(a) Relative error vs. amplitude")
ax.legend(fontsize=7, framealpha=0.7)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# (b) distribution (capped at 20%)
ax = axes[1]
ax.hist(np.clip(re_v, 0, 20), bins=100, color=COLOR_SC,
        edgecolor="none", alpha=0.85, density=True)
ax.axvline(mape,  color=COLOR_PRD, lw=1.2, ls="-",  label=f"MAPE = {mape:.1f}%")
ax.axvline(mdape, color=COLOR_RES, lw=1.2, ls="--", label=f"Median = {mdape:.1f}%")
ax.set_xlabel("Relative error (%, capped at 20%)")
ax.set_ylabel("Probability density")
ax.set_title("(b) Relative error distribution")
ax.legend(fontsize=7, framealpha=0.7)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# (c) MAPE per quartile
ax    = axes[2]
x_pos = np.arange(len(bin_stats))
w     = 0.38
b1    = ax.bar(x_pos - w/2, [b["mape"]  for b in bin_stats], w, color=COLOR_SC,  alpha=0.85, label="MAPE")
b2    = ax.bar(x_pos + w/2, [b["mdape"] for b in bin_stats], w, color=COLOR_RES, alpha=0.85, label="Median APE")
ax.set_xticks(x_pos)
ax.set_xticklabels([b["bin"].split("(")[0].strip() for b in bin_stats], fontsize=7)
ax.set_ylabel("Relative error (%)")
ax.set_title("(c) MAPE by amplitude quartile")
ax.legend(fontsize=7, framealpha=0.7)
ax.yaxis.set_minor_locator(AutoMinorLocator())
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=6.5)

fig6.tight_layout()
fig6.savefig("data/fig6_relative_error.png", dpi=200, bbox_inches="tight")

plt.show()
print("\nDone. All figures saved to data/")
print("  fig1_predicted_vs_actual.png")
print("  fig2_residual_vs_amplitude.png")
print("  fig3_error_distribution.png")
print("  fig4_waveform_zoom.png")
print("  fig5_loss_curves.png")
print("  fig6_relative_error.png")