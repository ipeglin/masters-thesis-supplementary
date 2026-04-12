"""
Fisher-Z Transform: Variance Stabilisation of Pearson Correlation Coefficients

The Pearson correlation coefficient r is bounded to [-1, 1]. Its sampling
variance depends on the true population correlation rho:

    Var(r) ≈ (1 - rho^2)^2 / (n - 1)

Near |rho| ≈ 1 the variance shrinks, so confidence intervals and hypothesis
tests that assume constant variance are invalid.

Fisher's Z-transform maps r to an approximately normally distributed statistic
with constant variance ≈ 1/(n-3):

    z = arctanh(r) = 0.5 * ln((1+r) / (1-r))

This script shows:
  1. The shape of the transformation and its derivative.
  2. Empirical sampling distributions of r and z at several true correlations.
  3. Empirical standard deviation of r and z as a function of rho.
  4. Effect of sample size on the Fisher-Z normal approximation quality.
"""

import os
from pathlib import Path

_matplotlib_cache_dir = Path(__file__).parent / ".cache" / "matplotlib"
_matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache_dir))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm

import plot_config  # noqa: F401

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_SAMPLES = 50       # sample size per simulated correlation estimate
N_SIMS = 10_000      # number of Monte-Carlo draws per true correlation
RNG_SEED = 42

RHO_SHOWCASE = [0.0, 0.3, 0.6, 0.9]        # true correlations to illustrate
RHO_GRID = np.linspace(-0.99, 0.99, 200)    # dense sweep for variance plot

RHO_FIXED = 0.8
SAMPLE_SIZES = [10, 20, 50, 100]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def simulate_correlations(rho: float, n: int, n_sims: int, rng) -> np.ndarray:
    """Return n_sims Pearson r estimates from bivariate normal with correlation rho."""
    cov = np.array([[1.0, rho], [rho, 1.0]])
    data = rng.multivariate_normal([0.0, 0.0], cov, size=(n_sims, n))  # (n_sims, n, 2)
    x, y = data[:, :, 0], data[:, :, 1]
    xm = x - x.mean(axis=1, keepdims=True)
    ym = y - y.mean(axis=1, keepdims=True)
    num = (xm * ym).sum(axis=1)
    denom = np.sqrt((xm ** 2).sum(axis=1) * (ym ** 2).sum(axis=1))
    return np.clip(num / denom, -0.9999, 0.9999)


# ---------------------------------------------------------------------------
# 1. The Fisher-Z mapping
# ---------------------------------------------------------------------------

def plot_mapping():
    r_vals = np.linspace(-0.999, 0.999, 1000)
    z_vals = np.arctanh(r_vals)
    deriv = 1.0 / (1.0 - r_vals ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(r_vals, z_vals, color="steelblue", linewidth=1.8)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel(r"Pearson $r$")
    ax.set_ylabel(r"Fisher $z = \operatorname{arctanh}(r)$")
    ax.set_title(r"Fisher-Z Mapping $r \to z$")
    ax.set_xlim(-1, 1)

    ax2 = axes[1]
    ax2.plot(r_vals, deriv, color="firebrick", linewidth=1.8)
    ax2.set_xlabel(r"Pearson $r$")
    ax2.set_ylabel(r"$dz/dr = 1/(1 - r^2)$")
    ax2.set_title("Derivative of the Transform")
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(0, 20)
    ax2.annotate(
        "Derivative grows\nnear $|r| \\to 1$,\nstretching the scale",
        xy=(0.85, 1.0 / (1 - 0.85 ** 2)),
        xytext=(0.3, 12),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Sampling distributions of r and z
# ---------------------------------------------------------------------------

def plot_sampling_distributions(sim_results: dict) -> plt.Figure:
    n_rho = len(RHO_SHOWCASE)
    fig, axes = plt.subplots(n_rho, 2, figsize=(11, 3.0 * n_rho), sharey=False)
    fig.suptitle(
        rf"Sampling Distributions of $r$ and $z = \operatorname{{arctanh}}(r)$"
        f"\n($n = {N_SAMPLES}$, {N_SIMS:,} simulations per $\\rho$)"
    )

    colors = plt.cm.tab10.colors

    for row, rho in enumerate(RHO_SHOWCASE):
        r_s = sim_results[rho]["r"]
        z_s = sim_results[rho]["z"]
        color = colors[row]

        ax_r = axes[row, 0]
        ax_r.hist(r_s, bins=60, density=True, color=color, alpha=0.65, edgecolor="none")
        ax_r.axvline(rho, color="black", linestyle="--", linewidth=1.2, label=rf"$\rho = {rho}$")
        ax_r.set_xlabel(r"Pearson $r$")
        ax_r.set_ylabel("Density")
        ax_r.set_title(rf"$r$ distribution  ($\rho = {rho}$)")
        ax_r.set_xlim(-1, 1)
        ax_r.legend(loc="upper left", fontsize=9)
        skew_r = float(np.mean(((r_s - r_s.mean()) / r_s.std()) ** 3))
        ax_r.text(
            0.97, 0.95, f"skewness = {skew_r:.2f}",
            transform=ax_r.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        )

        ax_z = axes[row, 1]
        ax_z.hist(z_s, bins=60, density=True, color=color, alpha=0.65, edgecolor="none")
        z_true = np.arctanh(rho) if abs(rho) < 1 else np.sign(rho) * 4.0
        z_std_theory = 1.0 / np.sqrt(N_SAMPLES - 3)
        x_theory = np.linspace(z_s.min() - 0.5, z_s.max() + 0.5, 400)
        ax_z.plot(
            x_theory,
            scipy_norm.pdf(x_theory, loc=z_true, scale=z_std_theory),
            color="black", linewidth=1.5, linestyle="--",
            label=rf"$\mathcal{{N}}(\operatorname{{arctanh}}(\rho),\,1/(n-3))$",
        )
        ax_z.axvline(z_true, color="black", linestyle=":", linewidth=1.0)
        ax_z.set_xlabel(r"Fisher $z$")
        ax_z.set_ylabel("Density")
        ax_z.set_title(rf"$z$ distribution  ($\rho = {rho}$)")
        ax_z.legend(loc="upper right", fontsize=9)
        skew_z = float(np.mean(((z_s - z_s.mean()) / z_s.std()) ** 3))
        ax_z.text(
            0.97, 0.05, f"skewness = {skew_z:.2f}",
            transform=ax_z.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Variance stabilisation sweep
# ---------------------------------------------------------------------------

def plot_variance_stabilisation(rng) -> plt.Figure:
    empirical_sd_r = np.empty(len(RHO_GRID))
    empirical_sd_z = np.empty(len(RHO_GRID))

    for i, rho in enumerate(RHO_GRID):
        r_s = simulate_correlations(rho, N_SAMPLES, N_SIMS, rng)
        empirical_sd_r[i] = r_s.std()
        empirical_sd_z[i] = np.arctanh(r_s).std()

    theory_sd_r = (1 - RHO_GRID ** 2) / np.sqrt(N_SAMPLES - 1)
    theory_sd_z = np.full_like(RHO_GRID, 1.0 / np.sqrt(N_SAMPLES - 3))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    fig.suptitle(
        rf"Standard Deviation vs True Correlation ($n = {N_SAMPLES}$, {N_SIMS:,} simulations)"
    )

    ax_r = axes[0]
    ax_r.plot(RHO_GRID, empirical_sd_r, color="steelblue", linewidth=1.5, label="Empirical SD$(r)$")
    ax_r.plot(RHO_GRID, theory_sd_r, color="steelblue", linewidth=1.5, linestyle="--",
              label=r"Theory: $(1-\rho^2)/\sqrt{n-1}$")
    ax_r.set_xlabel(r"True correlation $\rho$")
    ax_r.set_ylabel(r"Standard deviation of $r$")
    ax_r.set_title(r"Pearson $r$: variance depends on $\rho$")
    ax_r.legend(fontsize=9)
    ax_r.set_xlim(-1, 1)

    ax_z = axes[1]
    ax_z.plot(RHO_GRID, empirical_sd_z, color="firebrick", linewidth=1.5, label="Empirical SD$(z)$")
    ax_z.plot(RHO_GRID, theory_sd_z, color="firebrick", linewidth=1.5, linestyle="--",
              label=r"Theory: $1/\sqrt{n-3}$")
    ax_z.set_xlabel(r"True correlation $\rho$")
    ax_z.set_ylabel(r"Standard deviation of $z$")
    ax_z.set_title(r"Fisher $z$: variance is constant across $\rho$")
    ax_z.legend(fontsize=9)
    ax_z.set_xlim(-1, 1)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Effect of sample size on Fisher-Z approximation quality
# ---------------------------------------------------------------------------

def plot_sample_size_effect(rng) -> plt.Figure:
    fig, axes = plt.subplots(1, len(SAMPLE_SIZES), figsize=(13, 4.0), sharey=False)
    fig.suptitle(
        rf"Fisher-Z Distribution at $\rho = {RHO_FIXED}$ for Varying Sample Size"
        f" ({N_SIMS:,} simulations)"
    )

    z_true = np.arctanh(RHO_FIXED)

    for ax, n in zip(axes, SAMPLE_SIZES):
        r_s = simulate_correlations(RHO_FIXED, n, N_SIMS, rng)
        z_s = np.arctanh(r_s)

        z_std_theory = 1.0 / np.sqrt(n - 3) if n > 3 else 1.0
        x_theory = np.linspace(z_s.min() - 0.3, z_s.max() + 0.3, 400)

        ax.hist(z_s, bins=55, density=True, color="steelblue", alpha=0.6, edgecolor="none")
        ax.plot(
            x_theory,
            scipy_norm.pdf(x_theory, loc=z_true, scale=z_std_theory),
            color="black", linewidth=1.5, linestyle="--",
            label=rf"$\mathcal{{N}}(z_{{\rho}},\,1/(n-3))$",
        )
        ax.axvline(z_true, color="firebrick", linewidth=1.0, linestyle=":")
        ax.set_title(rf"$n = {n}$")
        ax.set_xlabel(r"Fisher $z$")
        ax.set_ylabel("Density" if ax is axes[0] else "")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "plots"
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    fig1 = plot_mapping()
    fig1.savefig(out_dir / "fisher_z_mapping.pdf")

    sim_results = {}
    for rho in RHO_SHOWCASE:
        r_samples = simulate_correlations(rho, N_SAMPLES, N_SIMS, rng)
        sim_results[rho] = {"r": r_samples, "z": np.arctanh(r_samples)}
    fig2 = plot_sampling_distributions(sim_results)
    fig2.savefig(out_dir / "fisher_z_sampling_distributions.pdf")

    fig3 = plot_variance_stabilisation(rng)
    fig3.savefig(out_dir / "fisher_z_variance_stabilisation.pdf")

    fig4 = plot_sample_size_effect(rng)
    fig4.savefig(out_dir / "fisher_z_sample_size_effect.pdf")

    plt.show()
