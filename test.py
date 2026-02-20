"""
test.py

This file contains multiple independent practice problems to help you learn
how to use the provided helper functions.

How to use:
1) Put this test.py in the SAME folder as the helper .py file they sent.
2) Change HELPERS_MODULE_NAME below to match your helper filename (without .py).
   Example: if the helper file is named "fhl_helpers.py", set:
       HELPERS_MODULE_NAME = "fhl_helpers"
3) Run:
   python test.py
"""

from __future__ import annotations

import importlib
import os
import numpy as np
import pandas as pd

# ===========================
# CHANGE THIS LINE ONLY
# ===========================
HELPERS_MODULE_NAME = "interview_prep"  # <-- rename to your helper file name (no .py)

H = importlib.import_module(HELPERS_MODULE_NAME)


# ---------------------------
# Small utilities for printing
# ---------------------------
def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def quick_report(res: dict, alpha: float = 0.05, topk: int = 5) -> None:
    """
    Prints a clean, interview-style summary of the chi-square result.
    """
    chi2 = res["chi2"]
    p = res["p_value"]
    df = res["df"]
    print(f"chi2={chi2:.6g} | df={df} | p_value={p:.6g} | alpha={alpha}")
    print("Decision:", "DIFFERENT (p < alpha)" if p < alpha else "SIMILAR (p >= alpha)")

    sm = res["summary"].copy()
    sm_sorted = sm.sort_values("chi2_contrib", ascending=False).head(topk)

    print("\nTop bins driving the difference:")
    cols = ["bin_low", "bin_high", "expected", "observed", "obs_minus_exp", "chi2_contrib"]
    print(sm_sorted[cols].to_string(index=False))

    print("\nMin expected after merging:", float(sm["expected"].min()))
    print("Number of final bins:", len(sm))


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# ---------------------------
# Global seed for reproducibility
# ---------------------------
np.random.seed(0)


# =============================================================================
# PROBLEM 1: "Perfect match" sample (should usually look SIMILAR)
# Goal: Sample is randomly drawn from population.
# =============================================================================
print_header("PROBLEM 1: Perfect match (random sample from population)")

pop_upb = np.random.lognormal(mean=np.log(180_000), sigma=0.55, size=50_000)
pop_upb = np.clip(pop_upb, 20_000, 1_500_000)  # keep range realistic-ish

sample_good = np.random.choice(pop_upb, size=2_000, replace=False)

res_good = H._chi_square_upb(pop_upb, sample_good, bin_strategy="fd", min_expected=5.0) # Freedman-Diaconis binning with min expected count of 5
quick_report(res_good, alpha=0.05)


# =============================================================================
# PROBLEM 2: "Biased high" sample (should usually look DIFFERENT)
# Goal: Sample has too many high-UPB values.
# =============================================================================
print_header("PROBLEM 2: Biased HIGH sample (too many large UPBs)")

cut = np.quantile(pop_upb, 0.80)  # top 20%
high_pool = pop_upb[pop_upb >= cut]
low_pool = pop_upb[pop_upb < cut]

sample_biased_high = np.concatenate([
    np.random.choice(high_pool, size=1400, replace=False),
    np.random.choice(low_pool, size=600, replace=False),
])

res_biased_high = H._chi_square_upb(pop_upb, sample_biased_high, bin_strategy="fd", min_expected=5.0)
quick_report(res_biased_high, alpha=0.05)


# =============================================================================
# PROBLEM 3: "Biased low" sample (should usually look DIFFERENT)
# Goal: Sample has too many low-UPB values.
# =============================================================================
print_header("PROBLEM 3: Biased LOW sample (too many small UPBs)")

sample_biased_low = np.concatenate([
    np.random.choice(low_pool, size=1400, replace=False),
    np.random.choice(high_pool, size=600, replace=False),
])

res_biased_low = H._chi_square_upb(pop_upb, sample_biased_low, bin_strategy="fd", min_expected=5.0)
quick_report(res_biased_low, alpha=0.05)


# =============================================================================
# PROBLEM 4: Use different binning strategies
# Goal: Compare FD vs Scott vs Sturges vs Quantile for the same data.
# =============================================================================
print_header("PROBLEM 4: Compare bin strategies on the SAME (biased high) sample")

strategies = [
    ("fd", {}),
    ("scott", {}),
    ("sturges", {}),
    ("quantile", {"n_bins": 20}),
]

for strat, kwargs in strategies:
    print("\n--- Strategy:", strat, "---")
    if strat == "quantile":
        res = H._chi_square_upb(pop_upb, sample_biased_high, bin_strategy=strat, n_bins=kwargs["n_bins"])
    else:
        res = H._chi_square_upb(pop_upb, sample_biased_high, bin_strategy=strat)

    quick_report(res, alpha=0.05, topk=3)


# =============================================================================
# PROBLEM 5: Show bins and histogram counts using the low-level helpers
# Goal: Learn what _make_bins_from_population and _hist_counts output.
# =============================================================================
print_header("PROBLEM 5: Inspect bins + histogram counts (low-level helpers)")

edges = H._make_bins_from_population(pop_upb, strategy="fd")
pop_counts = H._hist_counts(pop_upb, edges)
samp_counts = H._hist_counts(sample_good, edges)

print("Number of bins:", len(edges) - 1)
print("First 5 edges:", edges[:5])
print("Last 5 edges:", edges[-5:])
print("Population total counts:", pop_counts.sum())
print("Sample total counts:", samp_counts.sum())
print("First 10 population bin counts:", pop_counts[:10])
print("First 10 sample bin counts:", samp_counts[:10])


# =============================================================================
# PROBLEM 6: Force bin merging by adding extreme outliers (tail merging practice)
# Goal: See bins get merged due to tiny expected counts in tails.
# =============================================================================
print_header("PROBLEM 6: Force bin merging with extreme outliers")

pop_outliers = pop_upb.copy()
# Add a handful of huge values to create a very sparse upper tail bin
pop_outliers[:25] = np.linspace(2_000_000, 10_000_000, 25)

sample_from_outliers_pop = np.random.choice(pop_outliers, size=2_000, replace=False)

# Run full chi-square (merging happens inside)
res_outliers = H._chi_square_upb(pop_outliers, sample_from_outliers_pop, bin_strategy="fd", min_expected=5.0)
quick_report(res_outliers, alpha=0.05)

# Also demonstrate merging directly
edges2 = H._make_bins_from_population(pop_outliers, strategy="fd")
pop_counts2 = H._hist_counts(pop_outliers, edges2)
samp_counts2 = H._hist_counts(sample_from_outliers_pop, edges2)

exp2 = (pop_counts2 / pop_counts2.sum()) * samp_counts2.sum()
obs2 = samp_counts2.copy()

obs_m, exp_m, edges_m = H._merge_bins_tail_first(obs2, exp2, edges2, min_expected=5.0)
print("\nDirect merge demo:")
print("Bins BEFORE merge:", len(obs2))
print("Bins AFTER merge:", len(obs_m))
print("Min expected AFTER merge:", float(exp_m.min()))


# =============================================================================
# PROBLEM 7: Change the minimum expected threshold (5 vs 10)
# Goal: See how stricter rules cause more merging (fewer bins).
# =============================================================================
print_header("PROBLEM 7: Change min_expected (5 vs 10)")

res_min5 = H._chi_square_upb(pop_upb, sample_biased_high, bin_strategy="fd", min_expected=5.0)
res_min10 = H._chi_square_upb(pop_upb, sample_biased_high, bin_strategy="fd", min_expected=10.0)

print("\nmin_expected=5.0")
quick_report(res_min5, alpha=0.05, topk=3)

print("\nmin_expected=10.0")
quick_report(res_min10, alpha=0.05, topk=3)


# =============================================================================
# PROBLEM 8: Handle NaNs (robustness practice)
# Goal: Confirm NaNs are ignored and code still works.
# =============================================================================
print_header("PROBLEM 8: NaNs in data (robustness)")

pop_with_nans = pop_upb.copy()
idx = np.random.choice(len(pop_with_nans), size=300, replace=False)
pop_with_nans[idx] = np.nan

sample_with_nans = sample_good.copy().astype(float)
sample_with_nans[np.random.choice(len(sample_with_nans), size=20, replace=False)] = np.nan

res_nans = H._chi_square_upb(pop_with_nans, sample_with_nans, bin_strategy="fd", min_expected=5.0)
quick_report(res_nans, alpha=0.05)


# =============================================================================
# PROBLEM 9: Excel + plot function cvr_plot_pop_vs_sample
# Goal: Create a toy Excel file and run their plotting wrapper.
# NOTE: It only outputs a plot if p_value >= alpha.
# =============================================================================
print_header("PROBLEM 9: Excel input + cvr_plot_pop_vs_sample")

excel_path = "toy_case.xlsx"
ensure_dir(excel_path)

df_pop = pd.DataFrame({"UPB": pop_upb})
df_samp_good = pd.DataFrame({"UPB": sample_good})
df_samp_biased = pd.DataFrame({"UPB": sample_biased_high})

# Case A: good sample (often p >= 0.05 -> plot gets saved)
with pd.ExcelWriter(excel_path) as writer:
    df_pop.to_excel(writer, sheet_name="population", index=False)
    df_samp_good.to_excel(writer, sheet_name="sample", index=False)

print("Running plot for GOOD sample -> output image: plot_good.png (only if p>=alpha)")
H.cvr_plot_pop_vs_sample(excel_path, output_img="plot_good.png", alpha=0.05, show=False)

# Case B: biased sample (often p < 0.05 -> function returns without saving)
with pd.ExcelWriter(excel_path) as writer:
    df_pop.to_excel(writer, sheet_name="population", index=False)
    df_samp_biased.to_excel(writer, sheet_name="sample", index=False)

print("Running plot for BIASED sample -> output image: plot_biased.png (only if p>=alpha)")
H.cvr_plot_pop_vs_sample(excel_path, output_img="plot_biased.png", alpha=0.05, show=False)

print("\nDone. Check your folder for:")
print(" - toy_case.xlsx")
print(" - plot_good.png (maybe)")
print(" - plot_biased.png (maybe)")
print("\nReminder: Their plot function intentionally does NOTHING when p < alpha.")