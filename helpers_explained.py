from __future__ import annotations

"""
This file is a set of helper functions for a VERY SPECIFIC (but common) task:

    "Does my SAMPLE of loans look like the full POPULATION of loans
     (with respect to the UPB variable)?"

UPB is just a number per loan (often "Unpaid Principal Balance").
Population = all loans you have.
Sample     = the subset you selected for some purpose (review, modeling, validation, etc.).

If your sample has the "same shape" of UPB values as the population, that's good.
If your sample is skewed (too many big loans, too many small loans, etc.), that's bad.

This module:
1) creates bins ("buckets") from the POPULATION only
2) counts how many population + sample items land in each bin
3) computes what sample counts "should be" if sample truly matches the population
4) merges bins if some bins are too small (chi-square needs enough expected counts)
5) runs a chi-square goodness-of-fit test
6) optionally plots a bar chart comparing population vs sample proportions

Just remember:
- p-value < alpha (usually 0.05)  -> sample looks different (probably biased)
- p-value >= alpha               -> no strong evidence sample is different
"""

from pathlib import Path  # (not used directly below, but kept from original)
import numpy as np
import pandas as pd
from scipy.stats import chisquare
import matplotlib.pyplot as plt


# =============================================================================
# 1) BINNING: How do we create UPB "buckets"?
# =============================================================================

def _make_bins_from_population(pop_upb, strategy="fd", n_bins=20, fd_max_bins=100):
    """
    Build bin edges from POPULATION ONLY.

    Why population only?
    --------------------
    If you use sample to define bins, the sample is "influencing the test".
    That can hide problems or bias results.
    So we define bins using the population, then ask:
        "If the sample followed the population, how many sample points
         would fall into each bin?"

    What is "edges"?
    ---------------
    Histogram bin edges are numbers like:

        edges = [0, 50k, 100k, 200k, 500k]

    That creates bins:
        bin 0: [0, 50k)
        bin 1: [50k, 100k)
        bin 2: [100k, 200k)
        bin 3: [200k, 500k]

    Inputs
    ------
    pop_upb : array-like
        Population UPB values. May include NaN values.
    strategy : str OR list/array
        - "fd"       : Freedman-Diaconis rule (good default for skewed data)
        - "scott"    : Scott's rule (like FD but uses std deviation)
        - "sturges"  : simple rule based on log2(n)
        - "quantile" : bins are based on quantiles (each bin has similar population count)
        - OR you can pass a custom list/array of edges directly.
    n_bins : int
        Only used for "quantile" strategy. Example: n_bins=20 => 20 bins.
    fd_max_bins : int
        Cap on number of bins for FD/Scott so we don't create 1000 bins by accident.

    Returns
    -------
    edges : np.ndarray of shape (num_bins+1,)
        The bin boundaries.
    """

    # Convert input to a numpy array (so we can use numpy operations easily)
    pop_upb = np.asarray(pop_upb)

    # Drop NaNs (NaN = "not a number"). NaNs break percentile/stats calculations.
    pop_upb = pop_upb[~np.isnan(pop_upb)]

    # -------------------------
    # Case 1: user provided custom edges
    # -------------------------
    # If strategy is a list/tuple/array, treat it as "these are my bin edges"
    if isinstance(strategy, (list, tuple, np.ndarray)):
        edges = np.asarray(strategy, dtype=float)

        # Basic validation:
        # - Must be 1D
        # - Must have at least 3 values (so you get at least 2 bins)
        if edges.ndim != 1 or len(edges) < 3:
            raise ValueError("Custom edges must be 1D with at least 3 values.")
        return edges

    # -------------------------
    # Case 2: Freedman-Diaconis ("fd")
    # -------------------------
    # Intuition:
    # - Uses IQR (interquartile range) which is robust to outliers.
    # - Many financial variables are skewed/heavy-tailed, so FD often behaves well.
    #
    # FD bin width:
    #       h = 2 * IQR * n^(-1/3)
    #
    # Then:
    #       bins ≈ (max - min) / h
    #
    # Safeguards:
    # - If IQR is 0 (lots of duplicates), fallback to "sturges"
    # - If h <= 0, fallback to "sturges"
    if strategy == "fd":
        # 75th percentile and 25th percentile
        q75, q25 = np.percentile(pop_upb, [75, 25])
        iqr = q75 - q25  # interquartile range

        # If IQR=0, FD can't work (would create bin width 0)
        if iqr == 0:
            return _make_bins_from_population(pop_upb, strategy="sturges")

        # Compute the FD bin width
        h = 2 * iqr * (len(pop_upb) ** (-1 / 3))

        # If h is not positive, fallback
        if h <= 0:
            return _make_bins_from_population(pop_upb, strategy="sturges")

        # How many bins does that imply?
        bins = int(np.ceil((pop_upb.max() - pop_upb.min()) / h))

        # Clamp bins to a reasonable range so we don't create too many or too few.
        # - Minimum 5 bins (avoid being too coarse)
        # - Maximum fd_max_bins (avoid being too granular / slow / noisy)
        bins = int(np.clip(bins, 5, fd_max_bins))

        # Create evenly spaced edges from min to max
        return np.linspace(pop_upb.min(), pop_upb.max(), bins + 1)

    # -------------------------
    # Case 3: Scott's rule ("scott")
    # -------------------------
    # Similar spirit to FD but uses standard deviation sigma instead of IQR.
    #
    # Scott bin width:
    #       h = 3.5 * sigma * n^(-1/3)
    #
    # Safeguards:
    # - if n <= 1, can't compute meaningful sigma => fallback
    # - if sigma == 0, all values identical => fallback
    if strategy == "scott":
        n = len(pop_upb)
        if n <= 1:
            return _make_bins_from_population(pop_upb, strategy="sturges")

        # ddof=1 gives sample standard deviation (common statistical choice)
        sigma = np.std(pop_upb, ddof=1)
        if sigma == 0:
            return _make_bins_from_population(pop_upb, strategy="sturges")

        h = 3.5 * sigma * (n ** (-1 / 3))
        if h <= 0:
            return _make_bins_from_population(pop_upb, strategy="sturges")

        bins = int(np.ceil((pop_upb.max() - pop_upb.min()) / h))
        bins = int(np.clip(bins, 5, fd_max_bins))
        return np.linspace(pop_upb.min(), pop_upb.max(), bins + 1)

    # -------------------------
    # Case 4: Quantile bins ("quantile")
    # -------------------------
    # Intuition:
    # - If population is very skewed, equal-width bins can produce tiny tail bins.
    # - Quantile bins try to put about equal *population mass* in each bin.
    #
    # Example n_bins=4:
    # - edges at 0%, 25%, 50%, 75%, 100% quantiles
    #
    # Safeguards:
    # - if many ties => duplicate edges => remove duplicates
    # - if we end up with too few edges, fallback to "sturges"
    if strategy == "quantile":
        # qs = [0.0, 0.05, 0.10, ..., 1.0] if n_bins=20
        qs = np.linspace(0, 1, n_bins + 1)

        # Quantiles of population
        edges = np.quantile(pop_upb, qs)

        # Remove duplicate edges (happens if many values equal)
        edges = np.unique(edges)

        # Need at least 3 edges for at least 2 bins
        if len(edges) < 3:
            return _make_bins_from_population(pop_upb, strategy="sturges")
        return edges

    # -------------------------
    # Case 5: Sturges ("sturges")
    # -------------------------
    # Simple, old-school rule:
    #       bins = ceil(log2(n) + 1)
    #
    # This often creates fewer bins than FD/Scott.
    # Minimum of 5 bins is enforced to avoid super coarse binning.
    if strategy == "sturges":
        bins = int(np.ceil(np.log2(len(pop_upb)) + 1))
        bins = max(bins, 5)
        return np.linspace(pop_upb.min(), pop_upb.max(), bins + 1)

    # If user typed an unknown strategy string
    raise ValueError("Unknown strategy.")


# =============================================================================
# 2) HISTOGRAM COUNTS: Count how many values fall into each bin
# =============================================================================

def _hist_counts(x, edges, weights=None):
    """
    Histogram counts for x with given edges (ignores NaNs).

    This is basically:
        "Given the bins, how many numbers land in each bin?"

    Inputs
    ------
    x : array-like
        The data values (population or sample).
    edges : np.ndarray
        Bin edges created by _make_bins_from_population.
    weights : array-like or None
        Optional weights. If provided, each data point contributes its weight
        instead of "1". (Not used in the rest of this module by default.)

    Returns
    -------
    counts : np.ndarray of floats, length len(edges)-1
        counts[i] is how many values fell in bin i (or weighted sum).
    """

    x = np.asarray(x)

    # Ignore NaNs (can't bin NaNs)
    mask = ~np.isnan(x)
    x = x[mask]

    # If weights are provided, apply the same NaN mask so lengths match
    w = None if weights is None else np.asarray(weights)[mask]

    # numpy histogram gives counts for each bin
    counts, _ = np.histogram(x, bins=edges, weights=w)

    # Force float type (useful later when we do divisions)
    return counts.astype(float)


# =============================================================================
# 3) BIN MERGING: Ensure every expected bin count is big enough
# =============================================================================

def _merge_bins_tail_first(obs, exp, edges, min_expected):
    """
    Merge bins until all expected >= min_expected.

    Why is this necessary?
    ----------------------
    The chi-square test becomes unreliable if expected counts are too small.
    A common rule of thumb: each bin should have expected count >= 5.

    "Expected" here means:
        If the sample truly matches the population,
        how many sample items should fall in this bin?

    Inputs
    ------
    obs : array of observed counts (from sample)
    exp : array of expected counts (based on population proportions)
    edges : bin edges
    min_expected : float (e.g., 5.0)

    Returns
    -------
    obs, exp, edges after merging.
    Each merge reduces the number of bins by 1.
    """

    # Copy everything as float arrays so we can safely modify them
    obs = obs.astype(float).copy()
    exp = exp.astype(float).copy()
    edges = edges.astype(float).copy()

    # -------------------------------------------------------------------------
    # STEP A: Merge upper tail first
    # -------------------------------------------------------------------------
    # The upper tail is the "high UPB" end.
    # This is often sparse (few very large loans), so it's common to merge there.
    i = len(obs) - 1  # start from last bin
    while i >= 0 and len(obs) > 1:
        # If expected count is too small, merge bin i into bin i-1
        if exp[i] < min_expected:
            if i == 0:
                # Can't merge left if we're already at first bin
                break

            # Merge counts:
            # Add current bin into previous bin
            obs[i - 1] += obs[i]
            exp[i - 1] += exp[i]

            # Remove bin i from arrays
            obs = np.delete(obs, i)
            exp = np.delete(exp, i)

            # Remove the corresponding edge:
            # If we merge bin i into i-1, we remove the edge that separated them.
            edges = np.delete(edges, i)

            # Move one bin left
            i -= 1
        else:
            # If this bin is fine, move left
            i -= 1

    # -------------------------------------------------------------------------
    # STEP B: Merge lower tail next
    # -------------------------------------------------------------------------
    # Lower tail is the "low UPB" end (small loans).
    i = 0
    while i < len(obs) and len(obs) > 1:
        if exp[i] < min_expected:
            if i == len(obs) - 1:
                # Can't merge right if we're already at last bin
                break

            # Merge bin i into bin i+1 (to the right)
            obs[i + 1] += obs[i]
            exp[i + 1] += exp[i]

            # Delete bin i
            obs = np.delete(obs, i)
            exp = np.delete(exp, i)

            # If we merged i into i+1, remove edge i+1 (the boundary between them)
            edges = np.delete(edges, i + 1)

            # Note: we do NOT increment i here because after deletion,
            # the new bin at position i needs to be checked again.
        else:
            i += 1

    # -------------------------------------------------------------------------
    # STEP C: Safety loop
    # -------------------------------------------------------------------------
    # If there are STILL bins with exp < min_expected, fix them one-by-one.
    #
    # We repeatedly:
    # - find the bin with smallest expected count
    # - merge it with a neighbor
    #
    # This guarantees we end up with all bins meeting the threshold
    # (unless only 1 bin remains).
    while len(obs) > 1 and np.any(exp < min_expected):
        # k = index of smallest expected bin
        k = int(np.argmin(exp))

        if k == 0:
            # If smallest is the first bin, merge it into the next bin (bin 1)
            obs[1] += obs[0]
            exp[1] += exp[0]
            obs = np.delete(obs, 0)
            exp = np.delete(exp, 0)
            edges = np.delete(edges, 1)

        elif k == len(obs) - 1:
            # If smallest is last bin, merge it into previous bin
            obs[-2] += obs[-1]
            exp[-2] += exp[-1]
            obs = np.delete(obs, -1)
            exp = np.delete(exp, -1)
            edges = np.delete(edges, -1)

        else:
            # Middle bin: choose the neighbor with smaller expected count
            # (this tends to merge small bins together efficiently)
            j = k - 1 if exp[k - 1] <= exp[k + 1] else k + 1

            # Merge bins i1 and i2, and keep the merged result at the lower index
            i1, i2 = sorted([k, j])

            obs[i1] = obs[i1] + obs[i2]
            exp[i1] = exp[i1] + exp[i2]

            # Remove the other bin
            obs = np.delete(obs, i2)
            exp = np.delete(exp, i2)

            # Remove the edge between them
            edges = np.delete(edges, i2)

    return obs, exp, edges


# =============================================================================
# 4) MAIN TEST: Chi-square goodness-of-fit for sample vs population
# =============================================================================

def _chi_square_upb(pop_upb, sample_upb, *, bin_strategy="fd", n_bins=20, min_expected=5.0):
    """
    Core chi-square test returning stats dict and per-bin summary.

    What we are testing (in plain English)
    --------------------------------------
    We are checking whether the SAMPLE distribution of UPB matches
    the POPULATION distribution of UPB.

    Steps (simple):
    1) Make bins using population only
    2) Count population per bin
    3) Count sample per bin
    4) Convert population counts into "expected sample counts"
    5) Merge bins until expected counts are big enough
    6) Run chi-square test

    Expected sample count formula
    -----------------------------
    Let:
        pop_counts[i] = population count in bin i
        pop_total     = total population count
        sample_total  = total sample count

    Population proportion in bin i:
        pop_prop[i] = pop_counts[i] / pop_total

    Expected sample count in bin i if sample matches population:
        exp[i] = pop_prop[i] * sample_total

    Chi-square idea (no deep stats needed)
    -------------------------------------
    Compare observed sample counts (obs) to expected sample counts (exp).
    The test returns:
        chi2 statistic: how different overall
        p-value: how surprising this difference is if sample truly matches population

    Inputs
    ------
    pop_upb : array-like
    sample_upb : array-like
    bin_strategy : "fd" | "scott" | "sturges" | "quantile" | custom edges
    n_bins : int (for quantile)
    min_expected : float (default 5.0)

    Returns
    -------
    dict with:
      - chi2 : float
      - p_value : float
      - df : int (degrees of freedom)
      - summary : DataFrame with per-bin info
      - sample_size : float
      - population_size_equiv : float
      - bin_edges : final edges after merging
      - bin_strategy : chosen strategy
    """

    # 1) Build bin edges from population only
    edges = _make_bins_from_population(pop_upb, strategy=bin_strategy, n_bins=n_bins)

    # 2) Histogram counts for population and sample using the SAME edges
    pop_counts = _hist_counts(pop_upb, edges)
    samp_counts = _hist_counts(sample_upb, edges)

    # 3) Totals
    pop_total = pop_counts.sum()
    sample_total = samp_counts.sum()

    # If either group has no valid data, we can't do anything
    if pop_total == 0 or sample_total == 0:
        raise ValueError("No data for chi-square.")

    # 4) Compute expected sample counts based on population proportions
    exp = (pop_counts / pop_total) * sample_total

    # Observed counts are the sample histogram counts
    obs = samp_counts.copy()

    # 5) Merge bins until expected counts are large enough
    obs_m, exp_m, edges_m = _merge_bins_tail_first(obs, exp, edges, min_expected)

    # 6) Run Pearson chi-square goodness-of-fit test
    #
    # scipy.stats.chisquare compares f_obs vs f_exp and returns (chi2, p_value)
    chi2, p = chisquare(f_obs=obs_m, f_exp=exp_m)

    # Degrees of freedom for this goodness-of-fit is (#bins - 1)
    # (Because once you know counts in all but one bin, the last is determined by total.)
    df = len(obs_m) - 1

    # Per-bin chi-square contribution:
    # contribution_i = (obs_i - exp_i)^2 / exp_i
    #
    # If exp_i is 0, dividing would explode, so we use NaN there.
    contrib = (obs_m - exp_m) ** 2 / np.where(exp_m == 0, np.nan, exp_m)

    # Build a summary table that is VERY useful in interviews:
    # It lets you explain WHERE the mismatch is happening.
    summary = pd.DataFrame({
        # Each row is one final bin (after merging)
        "bin_low": edges_m[:-1],
        "bin_high": edges_m[1:],

        # Expected and observed counts
        "expected": exp_m,
        "observed": obs_m,

        # Difference (positive means sample has more than expected in this bin)
        "obs_minus_exp": obs_m - exp_m,

        # How much this bin contributes to the overall chi-square statistic
        "chi2_contrib": contrib,

        # Proportions (easier to interpret than raw counts sometimes)
        "pop_prop": exp_m / np.nansum(exp_m),
        "sample_prop": obs_m / np.nansum(obs_m),
    })

    # Return everything in a dictionary so you can easily print/use pieces
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


# =============================================================================
# 5) PLOTTING WRAPPER: Read Excel and plot pop vs sample proportions
# =============================================================================

def cvr_plot_pop_vs_sample(input_path, output_img="chi2_hist.png", alpha: float = 0.05, show: bool = True):
    """
    Plot grouped bars of population vs sample proportions (by final bins).

    This is a "wrapper" function meant for the case study format:
    - Read Excel file
    - Run the chi-square test
    - If sample looks similar (p >= alpha), output a plot

    IMPORTANT BEHAVIOR:
    -------------------
    This function ONLY creates the plot IF p_value >= alpha.

    If p_value < alpha, it returns immediately and produces NO image.
    Why? The author's convention is:
        "Only generate this chart when sample is acceptable / not significantly different."

    Inputs
    ------
    input_path : path to an Excel file
        Must contain:
          - sheet named "population" with a column "UPB"
          - sheet named "sample" with a column "UPB"
    output_img : filename to save plot to
    alpha : float
        Significance threshold, usually 0.05
    show : bool
        If True, show the plot window; if False, just save the file.

    Output
    ------
    Saves a PNG file (output_img) ONLY if p_value >= alpha.
    """

    # Read the Excel sheets.
    # They assume a specific format: sheet names are "population" and "sample".
    df_pop = pd.read_excel(input_path, sheet_name="population")
    df_sample = pd.read_excel(input_path, sheet_name="sample")

    # Convert the UPB column to numeric.
    # errors="coerce" means:
    #   - if value is not a number (like "abc"), turn it into NaN
    # Then later, NaNs will be ignored in histogram counting.
    pop = pd.to_numeric(df_pop["UPB"], errors="coerce").values
    samp = pd.to_numeric(df_sample["UPB"], errors="coerce").values

    # Run the main chi-square analysis
    res = _chi_square_upb(pop, samp)
    sm = res["summary"]

    chi2_val = res.get("chi2")
    p_val = res.get("p_value")

    # Only plot when p-value >= alpha (fail to reject that distributions match)
    if p_val is None or p_val < alpha:
        # Meaning:
        # - If p is small (< alpha), sample looks different -> no plot
        # - If p is missing (shouldn't happen normally), no plot
        return

    # We'll draw bars for each final bin.
    # x positions: 0..num_bins-1
    x = range(len(sm))

    # Create readable labels like "100000–150000"
    # (They cast bin edges to int for display.)
    labels = [f"{int(a)}–{int(b)}" for a, b in zip(sm["bin_low"], sm["bin_high"])]

    # Bar width controls spacing between population and sample bars
    width = 0.4

    # Start a new figure (chart)
    plt.figure(figsize=(12, 6))

    # Plot population proportions
    # i - width/2 shifts bars slightly left
    plt.bar(
        [i - width / 2 for i in x],
        sm["pop_prop"],
        width=width,
        label="Population",
        alpha=0.7
    )

    # Plot sample proportions
    # i + width/2 shifts bars slightly right
    plt.bar(
        [i + width / 2 for i in x],
        sm["sample_prop"],
        width=width,
        label="Sample",
        alpha=0.7
    )

    # Set x tick labels (rotate so they don't overlap)
    plt.xticks(list(x), labels, rotation=45, ha="right")

    # Axis labels and title (plain English)
    plt.ylabel("Proportion")
    plt.xlabel("UPB bins")
    plt.title("Population vs Sample UPB Distribution")

    # Legend shows which color is pop vs sample
    plt.legend()

    # Tight layout reduces label cut-off
    plt.tight_layout()

    # Add a text box in the top-left with test results.
    # This is nice for reporting / screenshots.
    txt = (
        f"$\\chi^2$ = {chi2_val:.4g}\n"
        f"$p$ = {p_val:.4g}\n"
        f"$\\alpha$ = {alpha:.4g}"
    )
    plt.gca().text(
        0.02, 0.98, txt,
        transform=plt.gca().transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8)
    )

    # Save the image file
    plt.savefig(output_img, dpi=150)

    # Optionally show the plot on screen
    if show:
        plt.show()

    # Close figure to avoid memory leaks when running repeatedly
    plt.close()