# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MAT_PATH = "data/compression_data_logdelay.mat"
REL_FLOOR_FRAC = 0.01
OUTDIR = "data/logdelay_results"
os.makedirs(OUTDIR, exist_ok=True)

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
COLOR_ACT = "#1a1a2e"
COLOR_PRD = "#e84545"
COLOR_CMP = "#2e86ab"
COLOR_RES = "#f18f01"


def metrics(y_true, y_pred):
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
    return np.abs(hilbert(np.ravel(x)))


def envelope_relative_error(env_true, y_true, y_pred, floor_frac=0.01):
    env_true = np.ravel(env_true)
    y_true   = np.ravel(y_true)
    y_pred   = np.ravel(y_pred)
    floor    = floor_frac * np.max(env_true)
    denom    = np.maximum(env_true, floor)
    rel_err  = np.abs(y_pred - y_true) / denom
    return rel_err, floor


# ══════════════════════════════════════════════════════════════════════════════
# LOAD + SPLIT
# ══════════════════════════════════════════════════════════════════════════════
sd = loadmat(MAT_PATH)
raw_in  = sd["train_input_real"].ravel()
raw_out = sd["train_output_real"].ravel()
N = min(len(raw_in), len(raw_out))

n_dev   = N // 2
n_train = int(0.8 * n_dev)

x_train_raw = raw_in[:n_train]
y_train     = raw_out[:n_train]
x_test_raw  = raw_in[n_dev:N]
y_test      = raw_out[n_dev:N]

# Normalized version for log/tanh/piecewise
x_scale = np.max(np.abs(x_train_raw))
x_train_n = x_train_raw / x_scale
x_test_n  = x_test_raw  / x_scale

env_test = analytic_envelope(y_test)

# Plot range: 95th percentile of |x|
x_plot_max = np.percentile(np.abs(x_train_raw), 95)

print(f"Train: {len(x_train_raw):,} samples")
print(f"Test:  {len(x_test_raw):,} samples")
print(f"Raw input range:  [{x_train_raw.min():.0f}, {x_train_raw.max():.0f}]")
print(f"Output range:     [{y_train.min():.0f}, {y_train.max():.0f}]")
print(f"Plot x limit (95th pct): {x_plot_max:.0f}")

ymax = np.max(np.abs(y_train))


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

# Log (normalized x in [-1, 1])
def fit_log(x_tr, y_tr):
    def model(x, a, b):
        return a * np.sign(x) * np.log1p(b * np.abs(x))
    try:
        popt, _ = curve_fit(model, x_tr, y_tr,
                            p0=[ymax / np.log(2), 1.0], maxfev=20000)
    except RuntimeError:
        popt = [ymax, 1.0]
    def predict(x_n): return model(x_n, *popt)
    return predict, "Log", popt


# Tanh (normalized x)
def fit_tanh(x_tr, y_tr):
    def model(x, a, b):
        return a * np.tanh(b * x)
    try:
        popt, _ = curve_fit(model, x_tr, y_tr,
                            p0=[ymax, 1.0], maxfev=20000)
    except RuntimeError:
        popt = [ymax, 1.0]
    def predict(x_n): return model(x_n, *popt)
    return predict, "Tanh", popt


# Odd poly fit on NORMALIZED x for numerical stability
def fit_odd_poly_norm(x_tr_n, y_tr, max_deg):
    powers = np.arange(1, max_deg + 1, 2)
    A = np.column_stack([x_tr_n ** p for p in powers])
    c, _, _, _ = np.linalg.lstsq(A, y_tr, rcond=None)
    def predict(x_n):
        A_t = np.column_stack([x_n ** p for p in powers])
        return A_t @ c
    return predict, f"Odd poly deg {max_deg}", c


# Piecewise linear (normalized x)
def fit_piecewise_linear(x_tr, y_tr, n_segments):
    ax = np.abs(x_tr)
    ay = np.abs(y_tr)
    same_sign = np.sign(x_tr) == np.sign(y_tr)
    ax_s = ax[same_sign]
    ay_s = ay[same_sign]

    breaks = np.percentile(ax_s, np.linspace(0, 100, n_segments + 1))
    breaks[0]  = 0.0
    breaks[-1] = np.max(ax) * 1.01

    y_nodes = [0.0]
    for i in range(n_segments):
        mask = (ax_s >= breaks[i]) & (ax_s < breaks[i + 1])
        y_nodes.append(np.mean(ay_s[mask]) if mask.sum() > 0 else y_nodes[-1])
    y_nodes = np.array(y_nodes)

    def predict(x_n):
        return np.sign(x_n) * np.interp(np.abs(x_n), breaks, y_nodes)

    return predict, f"Piecewise linear ({n_segments} seg)", {"breaks": breaks, "y_nodes": y_nodes}


# ══════════════════════════════════════════════════════════════════════════════
# FIT + EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FITTING MEMORYLESS INVERSE MODELS")
print("=" * 70)

# Each model gets evaluated on its appropriate test input (raw or normalized)
log_fn,  log_label,  _ = fit_log(x_train_n, y_train)
tanh_fn, tanh_label, _ = fit_tanh(x_train_n, y_train)
poly_fn, poly_label, _ = fit_odd_poly_norm(x_train_n, y_train, 11)
pw5_fn,  pw5_label,  _ = fit_piecewise_linear(x_train_n, y_train, 5)
pw10_fn, pw10_label, _ = fit_piecewise_linear(x_train_n, y_train, 10)
pw20_fn, pw20_label, _ = fit_piecewise_linear(x_train_n, y_train, 20)

# Each entry: (key, label, predict on appropriate test input, input type)
evals = [
    ("log",    log_label,  log_fn(x_test_n),     "norm"),
    ("tanh",   tanh_label, tanh_fn(x_test_n),    "norm"),
    ("poly11", poly_label, poly_fn(x_test_n),    "norm"),
    ("pw5",    pw5_label,  pw5_fn(x_test_n),     "norm"),
    ("pw10",   pw10_label, pw10_fn(x_test_n),    "norm"),
    ("pw20",   pw20_label, pw20_fn(x_test_n),    "norm"),
]

# Store the predict functions too for plotting later
predict_fns = {
    "log":    (log_fn,  "norm"),
    "tanh":   (tanh_fn, "norm"),
    "poly11": (poly_fn, "norm"),
    "pw5":    (pw5_fn,  "norm"),
    "pw10":   (pw10_fn, "norm"),
    "pw20":   (pw20_fn, "norm"),
}

print(f"\n  {'Model':<35} {'RMSE':>10} {'R²':>12} {'Env-MAPE':>10}")
print("  " + "-" * 70)

results = {}
for key, label, yhat, input_type in evals:
    if not np.all(np.isfinite(yhat)):
        print(f"  {label:<35} FAILED (NaN/Inf)")
        continue
    m   = metrics(y_test, yhat)
    rel, _ = envelope_relative_error(env_test, y_test, yhat, floor_frac=REL_FLOOR_FRAC)
    mape = 100 * rel.mean()
    print(f"  {label:<35} {m['RMSE']:>10,.2f} {m['R2']:>12.6f} {mape:>9.2f}%")
    results[key] = {
        'predict': predict_fns[key][0],
        'input_type': predict_fns[key][1],
        'label': label, 'yhat': yhat,
        'rmse': m['RMSE'], 'r2': m['R2'], 'mape': mape
    }

best_pw_key = max([k for k in results if k.startswith("pw")],
                  key=lambda k: results[k]['r2'])
best_all    = max(results.values(), key=lambda r: r['r2'])
print(f"\n  Best piecewise: {results[best_pw_key]['label']}  R²={results[best_pw_key]['r2']:.6f}")
print(f"  Best overall:   {best_all['label']}  R²={best_all['r2']:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
plot_keys = ["log", "tanh", "poly11", best_pw_key]
colors    = ["#f18f01", "#9b59b6", "#27ae60", COLOR_PRD]
styles    = ["--",      ":",       "-.",      "-"]
lws       = [1.4,       1.4,       1.4,       2.0]


# (a) scatter + curves, clipped to 95th percentile of |x|
fig, axes = plt.subplots(1, 2, figsize=(PANEL_2C + 0.4, PANEL_W + 0.4))


# (a) scatter + curves, clipped to 95th percentile of |x|
ax = axes[0]
rng = np.random.default_rng(42)
pos_mask = (x_train_raw > 0) & (x_train_raw <= x_plot_max)
sub = rng.choice(pos_mask.sum(), size=min(30000, pos_mask.sum()), replace=False)
ax.scatter(x_train_raw[pos_mask][sub], y_train[pos_mask][sub],
           s=0.5, alpha=0.08, color=COLOR_CMP, linewidths=0,
           rasterized=True, label="Data")

x_plot_raw = np.linspace(0, x_plot_max, 600)
x_plot_n   = x_plot_raw / x_scale

for i, key in enumerate(plot_keys):
    if key not in results:
        continue
    r = results[key]
    x_in = x_plot_raw if r['input_type'] == "raw" else x_plot_n
    y_plot = r['predict'](x_in)
    tag    = " (best)" if r is best_all else ""
    ax.plot(x_plot_raw, y_plot,
            color=colors[i], ls=styles[i], lw=lws[i],
            label=f"{r['label']}  R²={r['r2']:.3f}{tag}")

ax.set_xlim(0, x_plot_max)
ax.set_ylim(0, None)
ax.set_xlabel("Received sample $|x[n]|$ (a.u.)", fontsize=10)
ax.set_ylabel("Reference sample $|y[n]|$ (a.u.)", fontsize=10)
ax.set_title("(a) Transfer characteristic (positive half)", fontsize=9)
ax.tick_params(axis='both', labelsize=9)
ax.legend(fontsize=7, framealpha=0.7, loc="upper left")
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# (b) best reconstruction excerpt
ax  = axes[1]
exc = 60
yhat_best = best_all['yhat'][:exc]
ax.plot(np.arange(exc), y_test[:exc],  color=COLOR_ACT, lw=1.0, label="Reference")
ax.plot(np.arange(exc), yhat_best, color=COLOR_PRD, lw=1.0, ls=":",
        label=f"Best: {best_all['label']}")
ax.set_xlabel("Sample index", fontsize=10)
ax.set_ylabel("Amplitude (a.u.)", fontsize=10)
ax.set_title(f"(b) Reconstruction  R²={best_all['r2']:.3f}, Env-MAPE={best_all['mape']:.1f}%", fontsize=9)
ax.tick_params(axis='both', labelsize=9)
ax.legend(fontsize=8, framealpha=0.7)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# suptitle removed
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fig_memoryless_baseline.png"), dpi=200, bbox_inches="tight")
print(f"\nSaved: fig_memoryless_baseline.png")
plt.show()
print("\nDone!")