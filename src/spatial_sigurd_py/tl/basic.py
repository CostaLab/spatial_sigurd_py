import gc
from collections.abc import Iterable

import anndata as ad
import numpy as np
import pandas as pd

from spatial_sigurd_py._utils.basic import _strand_concordance


def Variant_Selection_VMR(
    adata,
    *,
    stabilize_variance: bool = True,
    low_coverage_threshold: int = 10,
    minimum_fw_rev_reads: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Python port of the R function `VariantSelection_VMR`.

    Parameters
    ----------
    adata
        AnnData returned by `load_mgatk_samples`.  Requires layers
        `"alt_reads"`, `"coverage"`, `"alt_fwd"`, `"alt_rev"`.
    stabilize_variance
        Fill low-coverage cells with the variant's mean AF (as in the
        original R code).
    low_coverage_threshold
        Coverage cut-off for the variance stabilisation step.
    minimum_fw_rev_reads
        Minimum counts (forward **and** reverse) to consider a cell
        “detected”.
    verbose
        Should the function give updates?

    Returns
    -------
    pandas.DataFrame
        One row per variant, with VMR, strand concordance, etc.
    """
    # 0. Helpers & aliases
    # These are parameters that are used in the function and I do not want to
    # supply as specific parameters. They are not important enough or can
    # immediately defined for all parts below.
    small_epsilon = 1e-11
    vnames = adata.var_names  # variants

    # 1. Basic matrices
    if verbose:
        print("Calculating basic matrices.")
    # end if statement

    alt_reads = adata.layers["alt_reads"].tocsc()
    coverage = adata.layers["coverage"].tocsc()
    n_cells = coverage.shape[0]

    # Per-variant allele frequency: sums reduce over cells (axis 0), since layers
    # are cells x variants. mean_af is length n_vars.
    sum_alts = np.asarray(alt_reads.sum(axis=0)).ravel().astype(np.float64)
    sum_coverage = np.asarray(coverage.sum(axis=0)).ravel().astype(np.float64)
    mean_af = np.divide(
        sum_alts,
        sum_coverage,
        out=np.zeros(alt_reads.shape[1], dtype=np.float64),
        where=sum_coverage > 0,
    )

    # Per-variant mean coverage (axis 0).
    mean_cov = np.asarray(coverage.mean(axis=0)).ravel()
    del sum_alts, sum_coverage

    # 2. Strand concordance, sparse-native, in variant order.
    if verbose:
        print("Computing strand concordance.")
    # end if statement

    fw_reads = adata.layers["alt_fwd"]
    rev_reads = adata.layers["alt_rev"]
    strand_concordance = pd.Series(
        _strand_concordance(fw_reads, rev_reads),
        index=vnames,
        dtype=np.float64,
    )

    # 3-4. Per-variant VAF variance with optional low-coverage stabilisation.
    # The stabilised fraction is dense by nature --- every cell below threshold
    # (the majority at HD scale, structural zeros included) is overwritten with
    # the variant's mean_af --- so it is never materialised. Instead the variance
    # is closed-form from moment sums: high-coverage cells keep their stored VAF,
    # the n_L low-coverage cells each contribute exactly mean_af.
    with np.errstate(divide="ignore", invalid="ignore"):
        cov_inv = coverage.astype(np.float64, copy=True)
        cov_inv.data = 1.0 / cov_inv.data
    # end with statement
    fraction = alt_reads.astype(np.float64).multiply(cov_inv).tocsc()

    if stabilize_variance:
        if verbose:
            print("Stabilising variance for low-coverage cells.")
        # end if statement

        mask_H = coverage >= low_coverage_threshold  # sparse; keeps VAF here
        n_H = np.asarray(mask_H.sum(axis=0)).ravel().astype(np.float64)
        n_L = n_cells - n_H  # take mean_af here

        frac_H = fraction.multiply(mask_H)
        S = np.asarray(frac_H.sum(axis=0)).ravel() + n_L * mean_af
        SS = np.asarray(frac_H.multiply(frac_H).sum(axis=0)).ravel() + n_L * mean_af * mean_af
    else:
        S = np.asarray(fraction.sum(axis=0)).ravel()
        SS = np.asarray(fraction.multiply(fraction).sum(axis=0)).ravel()
    # end if statement

    with np.errstate(divide="ignore", invalid="ignore"):
        variants_variance = (SS - S * S / n_cells) / (n_cells - 1)
    # end with statement
    del fraction

    # 5. Cells detected  (fw ≥ N **and** rev ≥ N)
    detected = (fw_reads >= minimum_fw_rev_reads).multiply(rev_reads >= minimum_fw_rev_reads)
    del fw_reads
    del rev_reads
    cells_detected = np.asarray(detected.sum(axis=0)).ravel()

    # 6.  Assemble the VOI table
    if verbose:
        print("Assembling final DataFrame.")
    # end if statement

    voi = pd.DataFrame(
        data={
            "variant": vnames,
            "vmr": variants_variance / (mean_af + small_epsilon),
            "vmr_log10": np.log10(variants_variance / (mean_af + small_epsilon)),
            "mean": mean_af.round(7),
            "variance": variants_variance.round(7),
            "cells_detected": cells_detected,
            "strand_correlation": strand_concordance,
            "mean_coverage": mean_cov,
        },
        index=vnames,
    )

    # memory-cleanup for large runs
    del coverage
    gc.collect()

    return voi


def _shannon(counts: np.ndarray, base: float = np.e) -> float:
    counts = counts.astype(float, copy=False)
    total = counts.sum()
    if total <= 0:
        return 0.0
    # end if statement

    p = counts / total
    p = p[p > 0]
    return -np.sum(p * (np.log(p) / np.log(base)))


def _simpson_index(counts: np.ndarray) -> float:
    # Matches vegan::diversity(index='simpson'): 1 - sum(p^2)
    counts = counts.astype(float, copy=False)
    total = counts.sum()
    if total <= 0:
        return 0.0
    # end if statement

    p = counts / total
    return 1.0 - np.sum(p * p)


def _inv_simpson(counts: np.ndarray) -> float:
    counts = counts.astype(float, copy=False)
    total = counts.sum()
    if total <= 0:
        return 0.0
    # end if statement

    p2_sum = np.sum((counts / total) ** 2)
    return (1.0 / p2_sum) if p2_sum > 0 else 0.0


def ClonalDiversity(
    adata: ad.AnnData | pd.DataFrame,
    grouping: str | None = "Clones",
    cells: Iterable[str] | None = None,
    diversity_measure: str = "EffectiveSpecies",
    base: float = np.e,
    verbose: bool = True,
) -> float:
    """
    Computation of clonal diversity using diversity measures.
    The clones are defined using the function ClonalDefinition.
    It is not required to use clones, but any column in the meta data can be used.

    Parameters
    ----------
    adata : AnnData or pandas.DataFrame
        If AnnData, uses adata.obs as the per-cell metadata. If DataFrame, it is
        treated directly as the per-cell metadata (index = cell IDs).
    grouping : str
        Column in obs/metadata to count clones/groups (default 'Clones').
    cells : iterable of str, optional
        Subset of cell IDs (obs index) to include. If None, use all cells.
    diversity_measure : {'EffectiveSpecies','shannon','simpson','invsimpson'}
    base : float
        Log base for Shannon and EffectiveSpecies (default: e).
    verbose : bool
        Print progress messages.

    Returns
    -------
    float
        Diversity value.
    """
    # Validate measure
    supported = {"EffectiveSpecies", "shannon", "simpson", "invsimpson"}
    if diversity_measure not in supported:
        raise ValueError(f"Your diversity measure '{diversity_measure}' is not supported.")
    # end if statement

    # Get obs dataframe
    if isinstance(adata, ad.AnnData):
        obs = adata.obs
    elif isinstance(adata, pd.DataFrame):
        obs = adata
    else:
        raise TypeError("Provide an AnnData or a pandas.DataFrame with per cell metadata.")
    # end if statement

    # Subset cells if requested
    if cells is not None:
        cells = list(cells)
        missing = [c for c in cells if c not in obs.index]
        if missing:
            raise ValueError(f"Not all your cells are present in the object. Missing (first 5): {missing[:5]}")
        # end if statement

        if verbose:
            print("Subsetting to requested cells.")
        # end if statement

        obs = obs.loc[cells]
    # end if statement

    # Check grouping column
    if grouping is None:
        raise ValueError("You must provide a 'grouping' column name.")
    # end if statement

    if grouping not in obs.columns:
        raise KeyError(f"Your grouping variable '{grouping}' is not in the object metadata.")
    # end if statement

    # Counts per group (like data.frame(table(clones)) in R)
    counts = obs[grouping].value_counts(dropna=False).to_numpy(dtype=float)

    # Compute diversity
    if diversity_measure == "shannon":
        return float(_shannon(counts, base=base))
    # end if statement

    if diversity_measure == "simpson":
        return float(_simpson_index(counts))
    # end if statement

    if diversity_measure == "invsimpson":
        return float(_inv_simpson(counts))
    # end if statement

    if diversity_measure == "EffectiveSpecies":
        H = _shannon(counts, base=base)
        return float(base**H)
    # end if statement

    # Shouldn't reach here
    raise RuntimeError("Unexpected branch in ClonalDiversity.")
