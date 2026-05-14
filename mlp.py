# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy.io import loadmat
from scipy.signal import hilbert
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
MAT_PATH = "data/compression_data_logdelay.mat"
WINDOW   = 16

# Best ensemble architectures
TOP_ARCHS = [
    (64, 32),
    (256, 128, 64),
    (128, 64, 32),
]

SHARED_PARAMS = dict(
    activation="tanh",
    solver="adam",
    alpha=5e-4,
    learning_rate_init=5e-4,
    max_iter=2500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=25,
)

REL_FLOOR_FRAC = 0.01  # Relative-error stability floor
EXCERPT_WIN = 80       # Overlay window length
OUTDIR = "data/logdelay_results"
os.makedirs(OUTDIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# JOURNAL STYLE
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
    "lines.linewidth":     1.0,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})

PANEL_W  = 3.3
PANEL_2C = 6.8

COLOR_ACT = "#1a1a2e"   # actual
COLOR_PRD = "#e84545"   # predicted
COLOR_CMP = "#2e86ab"   # compressed input
COLOR_RES = "#f18f01"   # residual / error

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def make_window_data(signal_in, signal_out, window):
    """Create sliding windows from input aligned with output targets."""
    a = signal_in.ravel()
    y = signal_out[:len(a)].reshape(-1, 1)
    a_padded = np.pad(a, (window - 1, 0), constant_values=0)
    X = sliding_window_view(a_padded, window_shape=window, axis=0)
    return X[:len(y)], y[:len(X)]


def prepare_data(mat_path, window):
    """Load and prepare training/validation/test splits with normalization."""
    sd = loadmat(mat_path)
    raw_in  = sd["train_input_real"].ravel()
    raw_out = sd["train_output_real"].ravel()

    X_raw, y_raw = make_window_data(raw_in, raw_out, window)

    N = len(X_raw)
    X_dev_raw,  y_dev_raw  = X_raw[:N // 2], y_raw[:N // 2]
    X_test_raw, y_test_raw = X_raw[N // 2:], y_raw[N // 2:]

    sv = int(0.8 * len(X_dev_raw))
    X_train_raw, y_train_raw = X_dev_raw[:sv], y_dev_raw[:sv]
    X_val_raw,   y_val_raw   = X_dev_raw[sv:], y_dev_raw[sv:]

    X_mean = X_train_raw.mean(0, keepdims=True)
    X_std  = X_train_raw.std(0, keepdims=True) + 1e-8
    y_mean = y_train_raw.mean(0, keepdims=True)
    y_std  = y_train_raw.std(0, keepdims=True) + 1e-8

    def sx(a): return (a - X_mean) / X_std
    def sy(a): return (a - y_mean) / y_std

    return {
        "X_train": sx(X_train_raw), "y_train": sy(y_train_raw),
        "X_val":   sx(X_val_raw),   "y_val":   sy(y_val_raw),
        "X_test":  sx(X_test_raw),  "y_test":  sy(y_test_raw),
        "X_test_raw": X_test_raw,   "y_test_raw": y_test_raw,
        "cmp_test": X_test_raw[:, -1].ravel(),
        "y_mean": y_mean, "y_std":  y_std,
    }


def unscale(y_s, d):
    """Unscale predictions back to original units."""
    return y_s * d["y_std"] + d["y_mean"]


def metrics(y_true, y_pred):
    """Calculate regression metrics."""
    yt = np.ravel(y_true)
    yp = np.ravel(y_pred)
    mse = mean_squared_error(yt, yp)
    return {
        "MSE":  mse,
        "RMSE": np.sqrt(mse),
        "MAE":  mean_absolute_error(yt, yp),
        "R2":   r2_score(yt, yp),
    }


def analytic_envelope(x):
    """Compute envelope via Hilbert transform."""
    return np.abs(hilbert(np.ravel(x)))


def envelope_relative_error(env_true, y_true, y_pred, floor_frac=0.01):
    """
    Relative error: |y_pred - y_true| / max(A_true, ε)
    where ε = floor_frac × max(A_true)
    """
    env_true = np.ravel(env_true)
    y_true   = np.ravel(y_true)
    y_pred   = np.ravel(y_pred)
    floor    = floor_frac * np.max(env_true)
    denom    = np.maximum(env_true, floor)
    rel_err  = np.abs(y_pred - y_true) / denom
    return rel_err, floor


def find_segment_by_envelope(env, lo, hi, win=80, n=500):
    """Find a window whose mean envelope falls in [lo, hi)."""
    step = max(1, len(env) // n)
    cands = [
        i for i in range(0, len(env) - win, step)
        if lo <= env[i:i + win].mean() < hi
    ]
    if not cands:
        all_means = [env[i:i + win].mean() for i in range(0, len(env) - win, step)]
        target = 0.5 * (lo + hi)
        cands = [np.argmin(np.abs(np.array(all_means) - target)) * step]
    return min(cands, key=lambda i: abs(env[i:i + win].mean() - 0.5 * (lo + hi)))


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("Training ensemble...")
data = prepare_data(MAT_PATH, WINDOW)

ens_preds = []
for i, arch in enumerate(TOP_ARCHS):
    print(f"  [{i+1}/{len(TOP_ARCHS)}] arch = {arch}")
    mlp = MLPRegressor(**SHARED_PARAMS, hidden_layer_sizes=arch, random_state=42 + i)
    mlp.fit(data["X_train"], np.ravel(data["y_train"]))
    yhat_s = mlp.predict(data["X_test"]).reshape(-1, 1)
    yhat = np.ravel(unscale(yhat_s, data))
    ens_preds.append(yhat)

yt  = np.ravel(unscale(data["y_test"], data))  # actual target
yp  = np.mean(ens_preds, axis=0)               # ensemble prediction
xc  = np.ravel(data["cmp_test"])               # compressed input
res = yt - yp
m   = metrics(yt, yp)

# Envelope analysis
env_true = analytic_envelope(yt)
env_pred = analytic_envelope(yp)
rel_err_env, rel_floor = envelope_relative_error(
    env_true, yt, yp, floor_frac=REL_FLOOR_FRAC
)
rel_err_pct = 100 * rel_err_env
mape_env = rel_err_pct.mean()

# Quartiles for overlay excerpts
q25, q50, q75 = np.percentile(env_true, [25, 50, 75])
qbins = [
    (0,   q25, "Low (Q1)"),
    (q25, q50, "Med-low (Q2)"),
    (q50, q75, "Med-high (Q3)"),
    (q75, env_true.max() * 1.01, "High (Q4)")
]

bin_stats = []
for lo, hi, label in qbins:
    mask = (env_true >= lo) & (env_true < hi)
    vals = rel_err_pct[mask]
    bin_stats.append({
        "bin": label,
        "n": int(mask.sum()),
        "mape": vals.mean() if len(vals) else np.nan,
        "mdape": np.median(vals) if len(vals) else np.nan,
        "p95": np.percentile(vals, 95) if len(vals) else np.nan,
    })

print("\n" + "=" * 65)
print("ENSEMBLE METRICS")
print("=" * 65)
print(f"  RMSE     : {m['RMSE']:>15,.2f}")
print(f"  MAE      : {m['MAE']:>15,.2f}")
print(f"  R²       : {m['R2']:>15.6f}")
print(f"  Env-MAPE : {mape_env:>14.2f}%")

# Sensitivity analysis for REL_FLOOR_FRAC
print("\n" + "=" * 65)
print("SENSITIVITY ANALYSIS: Relative Error Floor")
print("=" * 65)
floor_values = [0.005, 0.01, 0.02, 0.05]
print(f"  {'Floor %':<10} {'MAPE':<10} {'MdAPE':<10} {'P95':<10} {'P99':<10}")
print("  " + "-" * 50)
for floor_frac in floor_values:
    rel_test, _ = envelope_relative_error(env_true, yt, yp, floor_frac=floor_frac)
    rel_test_pct = 100 * rel_test
    print(f"  {floor_frac*100:>6.2f}%    "
          f"{rel_test_pct.mean():>8.2f}% "
          f"{np.median(rel_test_pct):>8.2f}% "
          f"{np.percentile(rel_test_pct, 95):>8.2f}% "
          f"{np.percentile(rel_test_pct, 99):>8.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Predicted vs. Actual Envelope
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(PANEL_W, PANEL_W))
ax.scatter(env_true, env_pred, s=1.5, alpha=0.2, color=COLOR_CMP, linewidths=0, rasterized=True)
elo, ehi = min(env_true.min(), env_pred.min()), max(env_true.max(), env_pred.max())
ax.plot([elo, ehi], [elo, ehi], "k--", lw=1.2, label=r"Ideal")
ax.set_xlabel(r"True envelope $A_{\mathrm{true}} = |\mathrm{hilbert}(y)|$ (a.u.)")
ax.set_ylabel(r"Predicted envelope $A_{\mathrm{pred}} = |\mathrm{hilbert}(\hat{y})|$ (a.u.)")
ax.set_title("Fig. 1 — Predicted vs. actual envelope amplitude")
ax.legend(loc="upper left", framealpha=0.7)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
fig1.tight_layout()
fig1.savefig(os.path.join(OUTDIR, "fig1_predicted_vs_actual_envelope.png"), dpi=200, bbox_inches="tight")
print("\nSaved: fig1_predicted_vs_actual_envelope.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Overlay Excerpts (Quartiles)
# ══════════════════════════════════════════════════════════════════════════════
starts = [
    find_segment_by_envelope(env_true, lo, hi, win=EXCERPT_WIN)
    for lo, hi, _ in qbins
]
qlabels = [row[2] for row in qbins]

fig2 = plt.figure(figsize=(PANEL_2C, PANEL_2C * 0.72))
gs = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.48, wspace=0.38)

for idx, (start, qlabel) in enumerate(zip(starts, qlabels)):
    ax = fig2.add_subplot(gs[idx // 2, idx % 2])
    t = np.arange(EXCERPT_WIN)

    xc_seg = xc[start:start + EXCERPT_WIN]
    yt_seg = yt[start:start + EXCERPT_WIN]
    yp_seg = yp[start:start + EXCERPT_WIN]

    ax.plot(t, xc_seg, color=COLOR_CMP, lw=1.1, ls=":", label="Received input", zorder=2)
    ax.plot(t, yt_seg, color=COLOR_ACT, lw=1.2, label="Reference target", zorder=4)
    ax.plot(t, yp_seg, color=COLOR_PRD, lw=1.2, ls="--", label="Reconstruction", zorder=5)

    ax.fill_between(t, yt_seg, yp_seg, alpha=0.12, color=COLOR_PRD, zorder=3)

    local_rmse = np.sqrt(np.mean((yt_seg - yp_seg) ** 2))
    local_env_true = analytic_envelope(yt_seg)
    local_rel, _ = envelope_relative_error(
        local_env_true, yt_seg, yp_seg, floor_frac=REL_FLOOR_FRAC
    )
    local_mape = 100 * np.mean(local_rel)

    letter = ["(a)", "(b)", "(c)", "(d)"][idx]
    ax.set_title(
        f"{letter} {qlabel}\n"
        f"RMSE = {local_rmse:,.0f}, Env-MAPE = {local_mape:.2f}%",
        pad=4
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if idx == 0:
        ax.legend(loc="upper right", framealpha=0.7, handlelength=1.8, fontsize=7)

fig2.suptitle(
    "Fig. 2 — Overlay excerpts across true-envelope amplitude quartiles",
    fontsize=9, fontweight="bold", y=1.01
)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTDIR, "fig2_overlay_excerpts.png"), dpi=200, bbox_inches="tight")
print("Saved: fig2_overlay_excerpts.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2b: Expanded View (showing non-linearity)
# ══════════════════════════════════════════════════════════════════════════════
# Scale the received input by 20x to show it's NOT just a linear scaling
# If reconstruction were linear, scaled input would match the target
expand_start = starts[0]
expand_len = 40

fig2b, ax = plt.subplots(figsize=(PANEL_W, PANEL_W * 0.75))
t_expand = np.arange(expand_len)

xc_exp = xc[expand_start:expand_start + expand_len]
yt_exp = yt[expand_start:expand_start + expand_len]
yp_exp = yp[expand_start:expand_start + expand_len]

# Scale the input to approximately match target amplitude range
scale_factor = 100
xc_scaled = xc_exp * scale_factor

ax.plot(t_expand, xc_exp, color=COLOR_CMP, lw=1.2, ls=":", label="Received input", marker='o', ms=3, zorder=2)
ax.plot(t_expand, xc_scaled, color='gray', lw=1.0, ls="--", alpha=0.6, label=f"Input × {scale_factor} (linear hypothesis)", marker='x', ms=2, zorder=1)
ax.plot(t_expand, yt_exp, color=COLOR_ACT, lw=1.2, label="Reference target", marker='s', ms=3, zorder=4)
ax.plot(t_expand, yp_exp, color=COLOR_PRD, lw=1.2, ls="--", label="Reconstruction", marker='^', ms=3, zorder=5)

# Annotate to show the difference
mid_pt = expand_len // 2
ax.annotate('', xy=(mid_pt, yt_exp[mid_pt]), xytext=(mid_pt, xc_scaled[mid_pt]),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.5, alpha=0.7))
ax.text(mid_pt + 2, (yt_exp[mid_pt] + xc_scaled[mid_pt]) / 2, 
        'Non-linear\ngap', fontsize=7, color='red', fontweight='bold')

ax.set_xlabel("Sample index")
ax.set_ylabel("Amplitude (a.u.)")
ax.set_title(f"Fig. 2b — Demonstrating non-linear reconstruction\n"
             f"(Scaled input × {scale_factor} does not match target)")
ax.legend(loc="upper right", framealpha=0.7, fontsize=6.5)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
fig2b.tight_layout()
fig2b.savefig(os.path.join(OUTDIR, "fig2b_expanded_nonlinear.png"), dpi=200, bbox_inches="tight")
print("Saved: fig2b_expanded_nonlinear.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Relative Error
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 3, figsize=(PANEL_2C, PANEL_2C * 0.42))
fig3.suptitle(
    r"Fig. 3 — Envelope-normalized relative error: "
    r"$|y_{\mathrm{pred}} - y_{\mathrm{true}}|\,/\,\max(A_{\mathrm{true}},\,\varepsilon)$",
    fontsize=9, fontweight="bold"
)

# (a) Relative error vs true envelope amplitude
ax = axes[0]
ax.scatter(env_true, rel_err_pct, s=1.2, alpha=0.15, color=COLOR_RES, linewidths=0, rasterized=True)

# Moving average
si = np.argsort(env_true)
xs = env_true[si]
rs = rel_err_pct[si]
win = max(1, len(xs) // 200)
kernel = np.ones(win) / win
roll = np.convolve(rs, kernel, mode="valid")
xmid = xs[win // 2 : win // 2 + len(roll)]

ax.plot(xmid, roll, color="k", lw=1.4, label="Moving avg.")
ax.axvline(rel_floor, color=COLOR_ACT, lw=1.2, ls="--", label="Amplitude floor $\\varepsilon$")
ax.set_xlabel("True envelope amplitude $A_{\\mathrm{true}}$ (a.u.)")
ax.set_ylabel("Relative error (%)")
ax.set_title("(a) Relative error vs. envelope amplitude")
ax.legend(fontsize=7, framealpha=0.7)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# (b) Distribution
ax = axes[1]
clip_hi = min(100, np.percentile(rel_err_pct, 99.5))
ax.hist(np.clip(rel_err_pct, 0, clip_hi), bins=100, color=COLOR_CMP, edgecolor="none", alpha=0.85, density=True)
mdape_env = np.median(rel_err_pct)
ax.axvline(mape_env,  color=COLOR_PRD, lw=1.2, ls="-",  label=f"MAPE = {mape_env:.1f}%")
ax.axvline(mdape_env, color=COLOR_RES, lw=1.2, ls="--", label=f"Median = {mdape_env:.1f}%")
ax.set_xlabel(f"Relative error (%, capped at {clip_hi:.1f})")
ax.set_ylabel("Probability density")
ax.set_title("(b) Distribution")
ax.legend(fontsize=7, framealpha=0.7)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# (c) Quartile bars
ax = axes[2]
x_pos = np.arange(len(bin_stats))
w = 0.38
b1 = ax.bar(x_pos - w / 2, [b["mape"] for b in bin_stats],  w, color=COLOR_CMP, alpha=0.85, label="MAPE")
b2 = ax.bar(x_pos + w / 2, [b["mdape"] for b in bin_stats], w, color=COLOR_RES, alpha=0.85, label="Median APE")

ax.set_xticks(x_pos)
ax.set_xticklabels([b["bin"].split("(")[0].strip() for b in bin_stats], fontsize=7)
ax.set_ylabel("Relative error (%)")
ax.set_title("(c) By envelope quartile")
ax.legend(fontsize=7, framealpha=0.7)
ax.yaxis.set_minor_locator(AutoMinorLocator())

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.1f}%", 
            ha="center", va="bottom", fontsize=6.5)

fig3.tight_layout()
fig3.savefig(os.path.join(OUTDIR, "fig3_relative_error.png"), dpi=200, bbox_inches="tight")
print("Saved: fig3_relative_error.png")

plt.show()
print("\nAll figures complete!")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE ON LOG DELAY DATA - BREAKDOWN BY WAVEFORM
# ══════════════════════════════════════════════════════════════════════════════
# compression_data_logdelay.mat IS the concatenation of eager1_log_delay_01 through _10
# So the train/test split is:
#   - First 50% of concatenated data (~waveforms 1-5) → train/val
#   - Second 50% (~waveforms 6-10) → hold-out test set
# 
# This section breaks down performance by individual waveform to see which ones
# are easier/harder for the model.
# ══════════════════════════════════════════════════════════════════════════════

import glob

log_delay_files = sorted(glob.glob("data/eager1_log_delay_*.mat"))
print(f"\n{'='*65}")
print(f"LOG DELAY TEST SET EVALUATION")
print(f"{'='*65}")
print(f"Found {len(log_delay_files)} log delay files")

if len(log_delay_files) > 0:
    # ══════════════════════════════════════════════════════════════════════════
    # WAVEFORM COMPARISON: How different are the 10 waveforms?
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("WAVEFORM STATISTICAL COMPARISON")
    print(f"{'='*65}")
    print(f"  {'File':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Env Mean':>10}")
    print("  " + "-" * 85)
    
    waveform_stats = []
    for f in log_delay_files:
        d = loadmat(f)
        sig_in = d["train_input_real"].ravel()
        sig_out = d["train_output_real"].ravel()
        env = analytic_envelope(sig_out)
        
        fname = os.path.basename(f)
        print(f"  {fname:<30} "
              f"{sig_out.mean():>10,.2f} "
              f"{sig_out.std():>10,.2f} "
              f"{sig_out.min():>10,.2f} "
              f"{sig_out.max():>10,.2f} "
              f"{env.mean():>10,.2f}")
        
        waveform_stats.append({
            'file': fname,
            'mean': sig_out.mean(),
            'std': sig_out.std(),
            'min': sig_out.min(),
            'max': sig_out.max(),
            'env_mean': env.mean()
        })
    
    # Summary statistics across waveforms
    df_stats = pd.DataFrame(waveform_stats)
    print("\n  Coefficient of Variation across waveforms:")
    print(f"    Mean:     {df_stats['mean'].std() / abs(df_stats['mean'].mean()) * 100:.2f}%")
    print(f"    Std:      {df_stats['std'].std() / df_stats['std'].mean() * 100:.2f}%")
    print(f"    Env Mean: {df_stats['env_mean'].std() / df_stats['env_mean'].mean() * 100:.2f}%")
    print(f"{'='*65}\n")
    # Concatenate all 10 waveforms
    all_in, all_out = [], []
    for f in log_delay_files:
        d = loadmat(f)
        all_in.append(d["train_input_real"].ravel())
        all_out.append(d["train_output_real"].ravel())

    raw_in_new  = np.concatenate(all_in)
    raw_out_new = np.concatenate(all_out)
    print(f"Total new test samples: {len(raw_in_new):,}")

    # Build windowed features
    X_new, y_new = make_window_data(raw_in_new, raw_out_new, WINDOW)

    # CRITICAL: Normalize using ORIGINAL training statistics
    # (NOT the statistics from this new data)
    X_train_mean = data["X_test_raw"].mean(0, keepdims=True)  # Use test as proxy for train stats
    X_train_std  = data["X_test_raw"].std(0, keepdims=True) + 1e-8
    X_new_s = (X_new - X_train_mean) / X_train_std

    # Actually, let's rebuild this properly from the data dict
    # We need to access the TRAINING normalization parameters
    # Let me recalculate from the prepare_data output
    
    # Reload to get proper training stats
    sd = loadmat(MAT_PATH)
    raw_in_orig  = sd["train_input_real"].ravel()
    raw_out_orig = sd["train_output_real"].ravel()
    X_raw_orig, y_raw_orig = make_window_data(raw_in_orig, raw_out_orig, WINDOW)
    N_orig = len(X_raw_orig)
    X_dev_orig = X_raw_orig[:N_orig // 2]
    sv_orig = int(0.8 * len(X_dev_orig))
    X_train_raw_orig = X_dev_orig[:sv_orig]
    
    X_mean_orig = X_train_raw_orig.mean(0, keepdims=True)
    X_std_orig  = X_train_raw_orig.std(0, keepdims=True) + 1e-8
    
    X_new_s = (X_new - X_mean_orig) / X_std_orig

    # Train new ensemble on the original data if we didn't save models
    # (Since we removed model saving in the cleanup)
    print("\nRetraining ensemble for log delay evaluation...")
    ens_models_new = []
    for i, arch in enumerate(TOP_ARCHS):
        mlp = MLPRegressor(**SHARED_PARAMS, hidden_layer_sizes=arch, random_state=42 + i)
        mlp.fit(data["X_train"], np.ravel(data["y_train"]))
        ens_models_new.append(mlp)

    # Ensemble predict on new data
    new_preds = []
    for mlp in ens_models_new:
        yhat_s = mlp.predict(X_new_s).reshape(-1, 1)
        new_preds.append(np.ravel(unscale(yhat_s, data)))

    yp_new = np.mean(new_preds, axis=0)
    yt_new = y_new.ravel()

    # Overall metrics
    m_new    = metrics(yt_new, yp_new)
    env_new  = analytic_envelope(yt_new)
    rel_new, _ = envelope_relative_error(
        env_new, yt_new, yp_new, floor_frac=REL_FLOOR_FRAC
    )

    print("\n" + "=" * 65)
    print("OVERALL METRICS (all 10 waveforms combined)")
    print("=" * 65)
    print(f"  RMSE     : {m_new['RMSE']:>15,.2f}")
    print(f"  MAE      : {m_new['MAE']:>15,.2f}")
    print(f"  R²       : {m_new['R2']:>15.6f}")
    print(f"  Env-MAPE : {100 * rel_new.mean():>14.2f}%")

    # Per-waveform breakdown
    print(f"\n{'='*65}")
    print("PER-WAVEFORM BREAKDOWN")
    print(f"{'='*65}")
    print(f"  {'File':<30} {'Samples':>8} {'RMSE':>10} {'R²':>10} {'Env-MAPE':>10}")
    print("  " + "-" * 75)
    
    offset = 0
    for f, arr_in in zip(log_delay_files, all_in):
        n     = len(arr_in)
        yt_w  = yt_new[offset:offset + n]
        yp_w  = yp_new[offset:offset + n]
        m_w   = metrics(yt_w, yp_w)
        env_w = analytic_envelope(yt_w)
        rel_w, _ = envelope_relative_error(env_w, yt_w, yp_w, floor_frac=REL_FLOOR_FRAC)
        fname = os.path.basename(f)
        print(
            f"  {fname:<30} "
            f"{n:>8,} "
            f"{m_w['RMSE']:>10,.2f} "
            f"{m_w['R2']:>10.6f} "
            f"{100 * rel_w.mean():>9.2f}%"
        )
        offset += n

else:
    print("\n⚠️  No log delay files found!")
    print("Expected files: data/eager1_log_delay_01.mat through _10.mat")
    print("\nTo copy from network drive, run this in MATLAB:")
    print("  src = 'Y:\\data\\Accum_Data\\2024_Antarctica_Ground2\\compression_measurements3\\';")
    print("  dst = 'C:\\Users\\s712a650\\Desktop\\project2.0\\decompression\\data\\';")
    print("  for i = 1:10")
    print("      fname = sprintf('eager1_log_delay_%02d.mat', i);")
    print("      copyfile(fullfile(src, fname), fullfile(dst, fname));")
    print("  end")

# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION: 5-Fold on individual waveforms
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("5-FOLD CROSS-VALIDATION (by waveform)")
print("=" * 65)
print("Training 5 folds, each using 8 waveforms for train, 2 for test...")

if len(log_delay_files) == 10:
    from sklearn.model_selection import KFold
    
    # Each fold: 8 waveforms train, 2 waveforms test
    fold_results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    waveform_arrays = [(all_in[i], all_out[i]) for i in range(10)]
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(waveform_arrays)):
        print(f"\n  Fold {fold_idx + 1}/5: Train on waveforms {train_idx + 1}, Test on waveforms {test_idx + 1}")
        
        # Concatenate training waveforms
        train_in = np.concatenate([waveform_arrays[i][0] for i in train_idx])
        train_out = np.concatenate([waveform_arrays[i][1] for i in train_idx])
        
        # Concatenate test waveforms
        test_in = np.concatenate([waveform_arrays[i][0] for i in test_idx])
        test_out = np.concatenate([waveform_arrays[i][1] for i in test_idx])
        
        # Create windowed data
        X_train_fold, y_train_fold = make_window_data(train_in, train_out, WINDOW)
        X_test_fold, y_test_fold = make_window_data(test_in, test_out, WINDOW)
        
        # Normalize
        X_mean_fold = X_train_fold.mean(0, keepdims=True)
        X_std_fold = X_train_fold.std(0, keepdims=True) + 1e-8
        y_mean_fold = y_train_fold.mean(0, keepdims=True)
        y_std_fold = y_train_fold.std(0, keepdims=True) + 1e-8
        
        X_train_s = (X_train_fold - X_mean_fold) / X_std_fold
        y_train_s = (y_train_fold - y_mean_fold) / y_std_fold
        X_test_s = (X_test_fold - X_mean_fold) / X_std_fold
        
        # Train ensemble for this fold
        fold_preds = []
        for arch in TOP_ARCHS:
            mlp = MLPRegressor(**SHARED_PARAMS, hidden_layer_sizes=arch, random_state=42)
            mlp.fit(X_train_s, np.ravel(y_train_s))
            yhat_s = mlp.predict(X_test_s).reshape(-1, 1)
            yhat = np.ravel(yhat_s * y_std_fold + y_mean_fold)
            fold_preds.append(yhat)
        
        # Ensemble prediction
        yp_fold = np.mean(fold_preds, axis=0)
        yt_fold = y_test_fold.ravel()
        
        # Metrics
        m_fold = metrics(yt_fold, yp_fold)
        env_fold = analytic_envelope(yt_fold)
        rel_fold, _ = envelope_relative_error(env_fold, yt_fold, yp_fold, floor_frac=REL_FLOOR_FRAC)
        
        fold_results.append({
            'fold': fold_idx + 1,
            'rmse': m_fold['RMSE'],
            'mae': m_fold['MAE'],
            'r2': m_fold['R2'],
            'env_mape': 100 * rel_fold.mean()
        })
        
        print(f"    RMSE: {m_fold['RMSE']:,.2f}, R²: {m_fold['R2']:.6f}, Env-MAPE: {100*rel_fold.mean():.2f}%")
    
    # Summary statistics
    df_cv = pd.DataFrame(fold_results)
    print("\n" + "=" * 65)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 65)
    print(f"  RMSE:     {df_cv['rmse'].mean():>10,.2f} ± {df_cv['rmse'].std():>8,.2f}")
    print(f"  MAE:      {df_cv['mae'].mean():>10,.2f} ± {df_cv['mae'].std():>8,.2f}")
    print(f"  R²:       {df_cv['r2'].mean():>10.6f} ± {df_cv['r2'].std():>8.6f}")
    print(f"  Env-MAPE: {df_cv['env_mape'].mean():>9.2f}% ± {df_cv['env_mape'].std():>7.2f}%")
    print("=" * 65)

else:
    print("Skipping cross-validation (need all 10 waveform files)")

print("\n" + "="*65)
print("DATA STRUCTURE EXPLANATION")
print("="*65)
print("compression_data_logdelay.mat = concatenation of all 10 waveforms")
print("\nTRAIN/TEST SPLIT:")
print("  First 50% of concatenated data → Train/Val (~waveforms 1-5)")
print("    ├─ 80% → Training set")
print("    └─ 20% → Validation set (early stopping)")
print("  Second 50% → Hold-out test set (~waveforms 6-10)")
print("\nWAVEFORM BREAKDOWN ABOVE:")
print("  - Waveforms 01-05: Were in training data")
print("  - Waveforms 06-10: Were in hold-out test set")
print("  - This breakdown shows which waveforms are easier/harder")
print("="*65)