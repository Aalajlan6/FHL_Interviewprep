from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chisquare
import matplotlib.pyplot as plt
# ----------------------------- Helpers ----------------------------- #
def _make_bins_from_population(pop_upb, strategy="fd", n_bins=20, fd_max_bins=100):
    """Build bin edges from population only."""
    pop_upb = np.asarray(pop_upb)
    pop_upb = pop_upb[~np.isnan(pop_upb)]

    if isinstance(strategy, (list, tuple, np.ndarray)):
        edges = np.asarray(strategy, dtype=float)
        if edges.ndim != 1 or len(edges) < 3:
            raise ValueError("Custom edges must be 1D with at least 3 values.")
        return edges
    
    # Freedman-Diaconis rule: bin width = 2 * IQR * n^(-1/3), with safeguards for small IQR or n.
    if strategy == "fd":
        q75, q25 = np.percentile(pop_upb, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return _make_bins_from_population(pop_upb, strategy="sturges")
        h = 2 * iqr * (len(pop_upb) ** (-1/3))
        if h <= 0:
            return _make_bins_from_population(pop_upb, strategy="sturges")
        bins = int(np.ceil((pop_upb.max() - pop_upb.min()) / h))
        bins = int(np.clip(bins, 5, fd_max_bins))
        return np.linspace(pop_upb.min(), pop_upb.max(), bins + 1)
    # Scott's rule is similar to Freedman-Diaconis but uses std dev instead of IQR, and a different constant.

    if strategy == "scott":
        n = len(pop_upb)
        if n <= 1:
            return _make_bins_from_population(pop_upb, strategy="sturges")
        sigma = np.std(pop_upb, ddof=1)
        if sigma == 0:
            return _make_bins_from_population(pop_upb, strategy="sturges")
        h = 3.5 * sigma * (n ** (-1/3))
        if h <= 0:
            return _make_bins_from_population(pop_upb, strategy="sturges")
        bins = int(np.ceil((pop_upb.max() - pop_upb.min()) / h))
        bins = int(np.clip(bins, 5, fd_max_bins))
        return np.linspace(pop_upb.min(), pop_upb.max(), bins + 1)
    
    # Quantile-based bins: edges at quantiles of the population distribution, with fallback to Sturges if too few unique edges.

    if strategy == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(pop_upb, qs)
        edges = np.unique(edges)
        if len(edges) < 3:
            return _make_bins_from_population(pop_upb, strategy="sturges")
        return edges
    
    # Sturges' formula: bins = ceil(log2(n) + 1), with a minimum of 5 bins to avoid too coarse binning for small samples.

    if strategy == "sturges":
        bins = int(np.ceil(np.log2(len(pop_upb)) + 1))
        bins = max(bins, 5)
        return np.linspace(pop_upb.min(), pop_upb.max(), bins + 1)

    raise ValueError("Unknown strategy.")

def _hist_counts(x, edges, weights=None):
    """Histogram counts for x with given edges (ignores NaNs)."""
    x = np.asarray(x)
    mask = ~np.isnan(x)
    x = x[mask]
    w = None if weights is None else np.asarray(weights)[mask]
    counts, _ = np.histogram(x, bins=edges, weights=w)
    return counts.astype(float)

def _merge_bins_tail_first(obs, exp, edges, min_expected):
    """Upper tail → lower tail → safety merges until all expected >= min_expected."""
    obs = obs.astype(float).copy()
    exp = exp.astype(float).copy()
    edges = edges.astype(float).copy()

    # Upper tail
    i = len(obs) - 1
    while i >= 0 and len(obs) > 1:
        if exp[i] < min_expected:
            if i == 0:
                break
            obs[i-1] += obs[i]; exp[i-1] += exp[i]
            obs = np.delete(obs, i); exp = np.delete(exp, i); edges = np.delete(edges, i)
            i -= 1
        else:
            i -= 1

    # Lower tail
    i = 0
    while i < len(obs) and len(obs) > 1:
        if exp[i] < min_expected:
            if i == len(obs) - 1:
                break
            obs[i+1] += obs[i]; exp[i+1] += exp[i]
            obs = np.delete(obs, i); exp = np.delete(exp, i); edges = np.delete(edges, i+1)
        else:
            i += 1

    # Safety
    while len(obs) > 1 and np.any(exp < min_expected):
        k = int(np.argmin(exp))
        if k == 0:
            obs[1] += obs[0]; exp[1] += exp[0]
            obs = np.delete(obs, 0); exp = np.delete(exp, 0); edges = np.delete(edges, 1)
        elif k == len(obs) - 1:
            obs[-2] += obs[-1]; exp[-2] += exp[-1]
            obs = np.delete(obs, -1); exp = np.delete(exp, -1); edges = np.delete(edges, -1)
        else:
            j = k - 1 if exp[k-1] <= exp[k+1] else k + 1
            i1, i2 = sorted([k, j])
            obs[i1] = obs[i1] + obs[i2]; exp[i1] = exp[i1] + exp[i2]
            obs = np.delete(obs, i2); exp = np.delete(exp, i2); edges = np.delete(edges, i2)
    return obs, exp, edges

def _chi_square_upb(pop_upb, sample_upb, *, bin_strategy="fd", n_bins=20, min_expected=5.0):
    """Core chi-square test returning stats dict and per-bin summary."""
    edges = _make_bins_from_population(pop_upb, strategy=bin_strategy, n_bins=n_bins)
    pop_counts = _hist_counts(pop_upb, edges)
    samp_counts = _hist_counts(sample_upb, edges)
    pop_total = pop_counts.sum(); sample_total = samp_counts.sum()
    if pop_total == 0 or sample_total == 0:
        raise ValueError("No data for chi-square.")
    exp = (pop_counts / pop_total) * sample_total
    obs = samp_counts.copy()
    obs_m, exp_m, edges_m = _merge_bins_tail_first(obs, exp, edges, min_expected)
    chi2, p = chisquare(f_obs=obs_m, f_exp=exp_m)
    df = len(obs_m) - 1
    contrib = (obs_m - exp_m) ** 2 / np.where(exp_m == 0, np.nan, exp_m)
    summary = pd.DataFrame({
        "bin_low": edges_m[:-1],
        "bin_high": edges_m[1:],
        "expected": exp_m,
        "observed": obs_m,
        "obs_minus_exp": obs_m - exp_m,
        "chi2_contrib": contrib,
        "pop_prop": exp_m / np.nansum(exp_m),
        "sample_prop": obs_m / np.nansum(obs_m),
    })
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "df": int(df),
        "summary": summary,
        "sample_size": float(sample_total),
        "population_size_equiv": float(pop_total),
        "bin_edges": edges_m,
        "bin_strategy": bin_strategy
    }

def cvr_plot_pop_vs_sample(input_path, output_img="chi2_hist.png", alpha: float = 0.05, show: bool = True):
    """Plot grouped bars of population vs sample proportions (by final bins)."""
    df_pop = pd.read_excel(input_path, sheet_name="population")
    df_sample = pd.read_excel(input_path, sheet_name="sample")
    pop = pd.to_numeric(df_pop["UPB"], errors="coerce").values
    samp = pd.to_numeric(df_sample["UPB"], errors="coerce").values
    res = _chi_square_upb(pop, samp)
    sm = res["summary"]

    chi2_val = res.get("chi2")
    p_val = res.get("p_value")
    # Only plot when p-value >= alpha (fail to reject H0)
    if p_val is None or p_val < alpha:
        return  # do not output plot under significant difference

    x = range(len(sm))
    labels = [f"{int(a)}–{int(b)}" for a, b in zip(sm["bin_low"], sm["bin_high"])]
    width = 0.4
    plt.figure(figsize=(12, 6))
    plt.bar([i - width/2 for i in x], sm["pop_prop"], width=width, label="Population", alpha=0.7)
    plt.bar([i + width/2 for i in x], sm["sample_prop"], width=width, label="Sample", alpha=0.7)
    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Proportion"); plt.xlabel("UPB bins")
    plt.title("Population vs Sample UPB Distribution")
    plt.legend(); plt.tight_layout()
    # annotate chi-square and p-value on the figure
    txt = f"$\\chi^2$ = {chi2_val:.4g}\n$p$ = {p_val:.4g}\n$\\alpha$ = {alpha:.4g}"
    plt.gca().text(0.02, 0.98, txt, transform=plt.gca().transAxes,
                   va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    plt.savefig(output_img, dpi=150)
    if show:
        plt.show()
    plt.close()
