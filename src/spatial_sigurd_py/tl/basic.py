import gc
from collections.abc import Iterable

import anndata as ad
import numpy as np
import pandas as pd

from spatial_sigurd_py._utils import _to_array


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
        Fill low-coverage cells with the variant’s mean AF (as in the
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
    n_vars = len(vnames)
    
    # 1. Basic matrices
    if verbose:
        print("Calculating basic matrices.")
    # end if statement
    
    alts = _to_array("alt_reads", adata_object=adata)  # variants × cells
    coverage = _to_array("coverage", adata_object=adata)
    fw_reads = _to_array("alt_fwd", adata_object=adata)
    rev_reads = _to_array("alt_rev", adata_object=adata)
    
    # Allele frequency per cell
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = alts / coverage
    
    fraction[np.isnan(fraction)] = 0.0
    
    # Means over *all* cells
    sum_alts = alts.sum(axis=1)  # per-variant due to 1
    sum_coverage = coverage.sum(axis=1)
    mean_af = np.divide(sum_alts, sum_coverage, out=np.zeros_like(sum_alts, dtype=np.float32), where=sum_coverage > 0)
    # Cleaning up a couple of unnecessary variables.
    del sum_coverage
    del sum_alts
    del alts
    mean_cov = np.array(coverage.mean(axis=1)).flatten()
    
    # 2. Strand concordance  (= Pearson between fw & rev per variant)
    # Each cell is an observation.
    if verbose:
        print("Computing strand concordance.")
    
    strand_corr = np.empty(n_vars, dtype=np.float32)
    for i in range(n_vars):
        fw = fw_reads[i]
        rv = rev_reads[i]
        mask = (fw > 0) | (rv > 0)  # ignore all-zero cells --> no signal
        if mask.sum() < 2:
            strand_corr[i] = np.nan
        else:
            strand_corr[i] = np.corrcoef(fw[mask], rv[mask])[0, 1]
    
    # 3. Optional variance stabilisation
    if stabilize_variance:
        if verbose:
            print("Stabilising variance for low-coverage cells.")
        
        low_cov_rows, low_cov_cols = np.where(coverage < low_coverage_threshold)
        # Replace the masked positions with the variant means.
        fraction[low_cov_rows, low_cov_cols] = mean_af[low_cov_rows]
        del low_cov_rows
    
    # 4. Row variance (ddof = 1 → sample variance, like MatrixGenerics)
    variants_variance = np.var(fraction, axis=1, ddof=1)
    del fraction
    
    # 5. Cells detected  (fw ≥ N **and** rev ≥ N)
    detected = (fw_reads >= minimum_fw_rev_reads) & (rev_reads >= minimum_fw_rev_reads)
    del fw_reads
    del rev_reads
    cells_detected = detected.sum(axis=1)
    
    # 6.  Assemble the VOI table
    if verbose:
        print("Assembling final DataFrame.")
    
    voi = pd.DataFrame(
        data={
            "variant": vnames,
            "vmr": variants_variance / (mean_af + small_epsilon),
            "vmr_log10": np.log10(variants_variance / (mean_af + small_epsilon)),
            "mean": mean_af.round(7),
            "variance": variants_variance.round(7),
            "cells_detected": cells_detected,
            "strand_correlation": strand_corr,
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
    
    p = counts / total
    p = p[p > 0]
    return -np.sum(p * (np.log(p) / np.log(base)))


def _simpson_index(counts: np.ndarray) -> float:
    # Matches vegan::diversity(index='simpson'): 1 - sum(p^2)
    counts = counts.astype(float, copy=False)
    total = counts.sum()
    if total <= 0:
        return 0.0
    
    p = counts / total
    return 1.0 - np.sum(p * p)


def _inv_simpson(counts: np.ndarray) -> float:
    counts = counts.astype(float, copy=False)
    total = counts.sum()
    if total <= 0:
        return 0.0
    
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
    
    # Get obs dataframe
    if isinstance(adata, ad.AnnData):
        obs = adata.obs
    elif isinstance(adata, pd.DataFrame):
        obs = adata
    else:
        raise TypeError("Provide an AnnData or a pandas.DataFrame with per cell metadata.")
    
    # Subset cells if requested
    if cells is not None:
        cells = list(cells)
        missing = [c for c in cells if c not in obs.index]
        if missing:
            raise ValueError(f"Not all your cells are present in the object. Missing (first 5): {missing[:5]}")
        
        if verbose:
            print("Subsetting to requested cells.")
        
        obs = obs.loc[cells]
    
    # Check grouping column
    if grouping is None:
        raise ValueError("You must provide a 'grouping' column name.")
    
    if grouping not in obs.columns:
        raise KeyError(f"Your grouping variable '{grouping}' is not in the object metadata.")
    
    # Counts per group (like data.frame(table(clones)) in R)
    counts = obs[grouping].value_counts(dropna=False).to_numpy(dtype=float)
    
    # Compute diversity
    if diversity_measure == "shannon":
        return float(_shannon(counts, base=base))
    
    if diversity_measure == "simpson":
        return float(_simpson_index(counts))
    
    if diversity_measure == "invsimpson":
        return float(_inv_simpson(counts))
    
    if diversity_measure == "EffectiveSpecies":
        H = _shannon(counts, base=base)
        return float(base**H)
    
    # Shouldn't reach here
    raise RuntimeError("Unexpected branch in ClonalDiversity.")
