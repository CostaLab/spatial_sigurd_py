import gc
import math
import traceback
import warnings
from collections.abc import Iterable, Sequence
from numbers import Integral, Real
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import SpaGFT as spg

from spatial_sigurd_py._exceptions import FTUWarning, SpotFilterWarning
from spatial_sigurd_py._utils import _safe_div_sparse, _strand_concordance

# These are the results from the FTU detection in SpaGFT.
# Since we run this on the enhanced and un-enhanced data,
# we want to save both results separately.
_FTU_RESULT_KEYS = (
    ("var", "ftu"),
    ("obsm", "ftu_binary"),
    ("obsm", "ftu_pseudo_expression"),
    ("uns", "detect_ftu_data"),
)


def _move_ftu_results(adata, suffix):
    """Rename SpaGFT's FTU outputs in place, tolerating missing keys."""
    for slot, key in _FTU_RESULT_KEYS:
        store = getattr(adata, slot)
        if key in store:
            store[f"{key}{suffix}"] = store[key]
            del store[key]


def _drop_ftu_results(adata):
    """Remove SpaGFT's FTU outputs, including partial state from a failed run."""
    for slot, key in _FTU_RESULT_KEYS:
        store = getattr(adata, slot)
        if key in store:
            del store[key]


_PCA_SIDE_EFFECT_KEYS = (
    ("obsm", "X_pca"),
    ("varm", "PCs"),
    ("uns", "pca"),
)


def _snapshot_slots(adata, slot_keys):
    """Remember the current value of every given slot that exists."""
    snapshot = {}
    for slot, key in slot_keys:
        store = getattr(adata, slot)
        if key in store:
            snapshot[(slot, key)] = store[key]
    return snapshot


def _restore_slots(adata, slot_keys, snapshot):
    """Reset the given slots to the snapshot, removing anything added since."""
    for slot, key in slot_keys:
        store = getattr(adata, slot)
        if key in store:
            del store[key]
        if (slot, key) in snapshot:
            store[key] = snapshot[(slot, key)]


def Variant_Selection_VMR(
    adata,
    *,
    stabilize_variance: bool = True,
    low_coverage_threshold: int | float = 10,
    minimum_fw_rev_reads: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Python port of the R function `VariantSelection_VMR`.

    Parameters
    ----------
    adata: AnnData
        AnnData returned by load_mgatk_samples.  Requires layers alt_reads, coverage, alt_fwd, alt_rev.
    stabilize_variance: bool
        Fill low-coverage cells with the variant's mean AF.
    low_coverage_threshold: int or float
        Coverage cut-off for the variance stabilisation step.
    minimum_fw_rev_reads: int
        Minimum counts (forward and reverse) to consider a cell detected.
    verbose: bool
        Should the function give updates?

    Returns
    -------
    pandas.DataFrame
        One row per variant, with VMR, strand concordance, etc.
    """
    if not isinstance(verbose, (bool, np.bool_)):
        raise TypeError(f"verbose must be bool. You supplied a {type(verbose).__name__}.")

    if not isinstance(stabilize_variance, (bool, np.bool_)):
        raise TypeError(f"stabilize_variance must be bool. You supplied a {type(stabilize_variance).__name__}.")

    if not isinstance(low_coverage_threshold, Real):
        raise TypeError(
            f"low_coverage_threshold must be an integer or a float. You supplied a {type(low_coverage_threshold).__name__}."
        )

    if isinstance(minimum_fw_rev_reads, (bool, np.bool_)):
        raise TypeError(
            f"minimum_fw_rev_reads must be an integer. You supplied a {type(minimum_fw_rev_reads).__name__}."
        )
    elif isinstance(minimum_fw_rev_reads, Integral):
        minimum_fw_rev_reads = int(minimum_fw_rev_reads)
    elif not isinstance(minimum_fw_rev_reads, Real):
        raise TypeError(
            f"minimum_fw_rev_reads must be an integer. You supplied a {type(minimum_fw_rev_reads).__name__}."
        )
    elif not float(minimum_fw_rev_reads).is_integer():
        raise ValueError(
            f"minimum_fw_rev_reads must be a whole number, since we are talking about reads. You supplied {minimum_fw_rev_reads!r}."
        )
    else:
        minimum_fw_rev_reads = int(minimum_fw_rev_reads)

    if minimum_fw_rev_reads < 1:
        raise ValueError(
            f"minimum_fw_rev_reads must be >= 1. We cannot have antimatter reads. You supplied {minimum_fw_rev_reads!r}."
        )

    # End of checks
    ####################################################################################################
    # 0. Helpers & aliases
    # These are parameters that are used in the function and I do not want to
    # supply as specific parameters. They are not important enough or can be
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
    # The stabilised fraction is dense by nature - every cell below threshold
    # (the majority at HD scale, structural zeros included) is overwritten with
    # the variant's mean_af  so it is never materialised. Instead the variance
    # is closed-form from moment sums: high-coverage cells keep their stored VAF,
    # the n_L low-coverage cells each contribute exactly mean_af.
    fraction = _safe_div_sparse(alt_reads, coverage, out_dtype=np.float64)

    if stabilize_variance and low_coverage_threshold > 0:
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

    # SS - S^2/n cancels catastrophically when (nearly) every cell holds the same value,
    # e.g. a variant whose cells are all below low_coverage_threshold. The result can then
    # be a tiny negative number, while the true variance is >= 0. NaN (n_cells == 1) stays NaN.
    variants_variance = np.maximum(variants_variance, 0.0)
    del fraction

    # 5. Cells detected  (fw >= N and rev >= N).
    if minimum_fw_rev_reads > 0:
        detected = (fw_reads >= minimum_fw_rev_reads).multiply(rev_reads >= minimum_fw_rev_reads)
        cells_detected = np.asarray(detected.sum(axis=0)).ravel()
    else:
        cells_detected = np.full(alt_reads.shape[1], n_cells, dtype=np.int64)

    del fw_reads
    del rev_reads

    # 6. Assemble the VOI table.
    if verbose:
        print("Assembling final DataFrame.")
    # end if statement

    vmr = variants_variance / (mean_af + small_epsilon)
    # A variant without variance has VMR 0, so its log10 is -inf. That is the correct value.
    with np.errstate(divide="ignore"):
        vmr_log10 = np.log10(vmr)
    # end with statement

    voi = pd.DataFrame(
        data={
            "variant": vnames,
            "vmr": vmr,
            "vmr_log10": vmr_log10,
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
    if not isinstance(verbose, (bool, np.bool_)):
        raise TypeError(f"verbose must be bool. You supplied a {type(verbose).__name__}.")

    # base is the base of a logarithm: log(1) = 0 would divide by zero, and log of a
    # non-positive number is undefined.
    if isinstance(base, (bool, np.bool_)) or not isinstance(base, Real):
        raise TypeError(f"base must be a float. You supplied a {type(base).__name__}.")
    elif not math.isfinite(base) or base <= 0 or base == 1:
        raise ValueError(f"base must be finite, > 0 and != 1, since it is a logarithm base. You supplied {base!r}.")

    # End checks
    ####################################################################################################
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

    if obs.index.has_duplicates:
        n_dup = int(obs.index.duplicated().sum())
        raise ValueError(
            f"The index contains {n_dup} duplicate cells, so counts per group would be wrong. Please make obs_names unique first."
        )

    # Subset cells if requested
    if cells is not None:
        # A single string is iterable, so list() would split it into characters.
        if isinstance(cells, (str, bytes)):
            raise TypeError(
                f"cells must be an iterable of cell IDs, not a single string. For one cell use [{cells!r}]."
            )
        # end if statement

        cells = list(cells)

        # obs.loc[cells] keeps repeated IDs, so they would be counted more than once.
        requested_cell_index = pd.Index(cells)
        if requested_cell_index.has_duplicates:
            number_of_duplicated_cells = int(requested_cell_index.duplicated().sum())
            raise ValueError(
                f"cells contains {number_of_duplicated_cells} duplicate cell IDs, which would be counted more than once."
            )
        # end if statement

        missing = [c for c in cells if c not in obs.index]
        if missing:
            raise ValueError(f"Not all your cells are present in the object. Missing (first 5): {missing[:5]}")
        # end if statement

        if verbose:
            print("Subsetting to requested cells.", flush=True)
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
    if verbose:
        n_dropped = int(obs[grouping].isna().sum())
        if n_dropped:
            print(
                f"Excluding {n_dropped} of {len(obs)} cells with no {grouping} assignment.",
                flush=True,
            )

    counts = obs[grouping].value_counts(dropna=True).to_numpy(dtype=float)

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
        # Without any assigned cell there is no species. base**0 would report one,
        # while the other measures return 0 here.
        if counts.sum() <= 0:
            return 0.0
        # end if statement

        H = _shannon(counts, base=base)
        return float(base**H)
    # end if statement

    # This should not happen.
    raise RuntimeError("Unexpected branch in ClonalDiversity.")


def Spatial_Clone_Detection(
    adata,
    spatial_key: str = "spatial",
    filter_by_min_spots: bool | int | float = 5,
    normalize: bool = True,
    neighbors_for_ratio: float | int | str = 8,
    neighbors_for_ratio_absolute: bool = True,
    detect_svg_normalize_lap: bool = False,
    detect_svg_filter_peaks: bool = True,
    detect_svg_S: int | float = 6,
    detect_svg_cal_pval: bool = True,
    p_value_threshold_svvs: float = 0.05,
    perform_low_pass_enhancement: bool = True,
    low_pass_enhancement_ratio_low_freq: float | int | str = "infer",
    low_pass_enhancement_c: float = 0.001,
    low_pass_enhancement_normalize_lap: bool = False,
    perform_ftu: bool = True,
    ftu_resolution=(0.7, 1.8, 0.1),
    ftu_weight_by_freq: bool = False,
    ftu_normalize_lap: bool = False,
    neighbors_for_ftu: int = 15,
    ftu_random_state: int = 0,
    ftu_on_error: Literal["warn", "raise"] = "warn",
    verbose: bool = True,
):
    """
    Detection of spatial clones using spatial graph Fourier transformation.
    This function is a wrapper around functions from SpaGFT.
    The function takes as input an AnnData object and processes it to detect
    Spatially Variable Variants (SVVs). These are then returned as a list in addition
    to the AnnData object.


    Parameters
    ----------
    adata: AnnData
        The AnnData object with coordinates
    spatial_key: str
        name of the spatial slot in adata.obsm
    filter_by_min_spots: bool, int or float
        The minimum number of spots that should have a variant.
        It can be False, an integer or a float (converted to int).
    normalize: bool
        Should the data in X be normalized (log1p) or used as is?
    neighbors_for_ratio: int, float or str
        The number of neighbors used. Either a float, integer or infer.
        If neighbors_for_ratio_absolute is True, it must be an integer.
    neighbors_for_ratio_absolute: bool
        Do you want a specific number of spots to be used for the neighborhood?
        If True, neighbors_for_ratio must be an integer (the number of spots).
        If False, the value of neighbors_for_ratio is used as the ratio.
    detect_svg_normalize_lap: bool
        Should the laplacian matrix be normalized?
    detect_svg_filter_peaks: bool
        Do you want to filter low peaks from the frequency domains?
    detect_svg_S: int or float
        The sensitivity of the Kneedle algorithm. Higher S means more variants are considered.
    detect_svg_cal_pval: bool
        Should P values be calculated for the SVVs?
        If yes, you can also subset them according to p_value_threshold_svvs.
        P values are corrected using the Benjamini-Yekutieli correction.
    p_value_threshold_svvs: float
        Significance threshold for the P value.
    perform_low_pass_enhancement: bool
        Should the data be enhanced?
    low_pass_enhancement_ratio_low_freq: float, int, str
        ratio_low_freq determines the number of the Fourier modes with low frequencies.
        ratio_low_freq * sqrt(number of spots) low frequency Fourier modes.
        infer uses heuristics.
    low_pass_enhancement_c: float
        c balances the smoothness and difference with previous expression.
        A high c leads to higher smoothness.
        c should be set to [0, 0.1]. The default is 0.001.
    low_pass_enhancement_normalize_lap: bool
        Should the laplacian matrix be normalized?
    perform_ftu: bool
        Should the variants be organized in units using the Louvain algorithm?
    ftu_resolution : float | list | tuple
        Resolution of the Louvain clustering that groups SVVs.
        A float means that only this resolution is used.
        A list is treated as a set of resolutions, which should all be floats.
        A tuple should be (start, end, step). This is the default.
    ftu_weight_by_freq: bool
        Should the Fourier components be weighted according to their frequency?
    ftu_normalize_lap: bool
        Should the laplacian matrix be normalized?
    neighbors_for_ftu: int
        Neighbors for the Louvain algorithm.
    ftu_random_state: int
        Random state for the Louvain algorithm.
    ftu_on_error: Literal["warn", "raise"]
        Error or warning for FTU detection.
    verbose: bool
        Should the function be verbose?

    Returns
    -------
    Returns a 4-tuple. The last element is None if perform_ftu is False
    adata: AnnData
        The modified input object. The spots are reordered, which ensures consistent
        results. Spots were reordered by the second coordinate column of
        adata.obsm[spatial_key], then by the first. adata.obs is not modified;
        SpaGFT reads the coordinates directly from adata.obsm[spatial_key].
        Variants might have been filtered and the data might have been normalized.
        If low pass enhancement was performed, X is overwritten with the enhanced
        signal. The original values are saved in the layer X_original.
        When the SVVs are organized using SpaGFT's FTU assignments, this is also
        added to the object.
    variant_df: pandas.DataFrame
        Statistics for each SVV, not subset.
        If FTUs are detected, it is subset.
        Only contains P values if detect_svg_cal_pval is True.
    svg_list: list of str, variant names
        Spatially variable variants selected from variant_df.
    variant_df_enhanced: pandas.DataFrame or None
        Same as variant_df but based on the enhanced signal.
    """
    if not isinstance(verbose, (bool, np.bool_)):
        raise TypeError(f"verbose must be bool. You supplied a {type(verbose).__name__}.")

    # Checking if min_spots is actually a valid integer.
    if isinstance(filter_by_min_spots, (bool, np.bool_)):
        if filter_by_min_spots:
            raise TypeError("filter_by_min_spots cannot be True. Use int, float, or False.")
        min_spots = None
    elif isinstance(filter_by_min_spots, Integral):
        min_spots = int(filter_by_min_spots)
        filter_by_min_spots = True
    elif isinstance(filter_by_min_spots, Real):
        value = float(filter_by_min_spots)
        if not math.isfinite(value):
            raise ValueError(f"filter_by_min_spots must be finite. You supplied {filter_by_min_spots!r}.")
        min_spots = math.ceil(value)
        if verbose:
            print(f"filter_by_min_spots is a float ({value}). ", flush=True)
            print(f"Rounding up to {min_spots}.", flush=True)
        filter_by_min_spots = True
    else:
        raise TypeError(
            f"filter_by_min_spots must be bool, int, or float. You supplied {type(filter_by_min_spots).__name__}."
        )

    if filter_by_min_spots and min_spots < 1:
        raise ValueError(f"filter_by_min_spots must be >= 1 when filtering. You supplied {min_spots}.")
    elif filter_by_min_spots and min_spots > adata.shape[0]:
        raise ValueError(
            f"filter_by_min_spots must be <= the number of spots when filtering. You supplied {min_spots}."
        )
    elif filter_by_min_spots and min_spots == adata.shape[0]:
        warnings.warn(
            f"filter_by_min_spots ({min_spots}) equals the number of spots. Only features detected in every single spot will be kept, which will probably return an empty object.",
            SpotFilterWarning,
            stacklevel=2,
        )

    # We check if the spatial_key is valid and where the spatial information is located.
    if not isinstance(spatial_key, str):
        raise TypeError(f"spatial_key must be a string. You supplied a {type(spatial_key).__name__}.")

    if spatial_key not in adata.obsm:
        available = list(adata.obsm.keys())
        raise KeyError(
            f"No spatial coordinates found under adata.obsm[{spatial_key!r}]. Available keys are: {available}."
        )

    # We check if p_value_threshold_svvs is float between 0 and 1.
    if isinstance(p_value_threshold_svvs, (bool, np.bool_)):
        raise TypeError(f"p_value_threshold_svvs must be float. You supplied {type(p_value_threshold_svvs).__name__}.")
    elif not isinstance(p_value_threshold_svvs, Real):
        raise TypeError(f"p_value_threshold_svvs must be float. You supplied {type(p_value_threshold_svvs).__name__}.")
    elif p_value_threshold_svvs > 1 or p_value_threshold_svvs <= 0:
        raise ValueError(f"p_value_threshold_svvs must be 0 < x <= 1. You supplied {p_value_threshold_svvs}.")

    coords = np.asarray(adata.obsm[spatial_key])
    if coords.ndim != 2:
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must be a 2D array. You have {coords.ndim} dimensions with the shape {coords.shape}."
        )
    if coords.shape[1] != 2:
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must have exactly 2 columns. You have {coords.shape[1]} columns."
        )

    if not np.issubdtype(coords.dtype, np.number):
        raise TypeError(f"adata.obsm[{spatial_key!r}] must be numeric. You have a dtype of {coords.dtype}.")

    coords = coords.astype(float, copy=False)

    if not np.isfinite(coords).all():
        n_bad = int((~np.isfinite(coords)).any(axis=1).sum())
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] contains non-finite values in {n_bad} of {coords.shape[0]} spots."
        )

    if not isinstance(neighbors_for_ratio_absolute, (bool, np.bool_)):
        raise TypeError(
            f"neighbors_for_ratio_absolute must be bool. You supplied a {type(neighbors_for_ratio_absolute).__name__}."
        )

    if isinstance(neighbors_for_ratio, str):
        if neighbors_for_ratio != "infer":
            raise ValueError(
                f"neighbors_for_ratio must be an integer, float, or infer. You supplied {neighbors_for_ratio!r}."
            )
        if neighbors_for_ratio_absolute:
            raise TypeError(
                f"If neighbors_for_ratio_absolute is True, neighbors_for_ratio must be an integer. You supplied {neighbors_for_ratio}."
            )
        # While infer is valid for the functions, we use 1 to stay consistent with detect_svg.
        # Please note that identify_ftu uses 2.0.
        ratio_use = 1.0
    elif isinstance(neighbors_for_ratio, (bool, np.bool_)):
        raise TypeError("neighbors_for_ratio must either be an integer, float or infer. You supplied a bool.")
    elif isinstance(neighbors_for_ratio, Integral):
        if neighbors_for_ratio_absolute:
            # We want to see the results if we use a KNN with a specific K.
            if verbose:
                print(f"Determining the ratio to use for {neighbors_for_ratio} neighbors.", flush=True)
            ratio_use = neighbors_for_ratio / np.ceil(np.sqrt(adata.shape[0]) / 2)
        else:
            # We want to use the integer as a ratio, not as the specific number of cells.
            ratio_use = float(neighbors_for_ratio)
    elif isinstance(neighbors_for_ratio, Real):
        if neighbors_for_ratio_absolute:
            raise TypeError(
                f"If neighbors_for_ratio_absolute is True, neighbors_for_ratio must be an integer. You supplied {neighbors_for_ratio}."
            )
        ratio_use = float(neighbors_for_ratio)
        if not math.isfinite(ratio_use):
            raise ValueError(f"neighbors_for_ratio must be finite. You supplied {neighbors_for_ratio!r}.")
    else:
        raise TypeError(
            f"neighbors_for_ratio must either be an integer, float or infer. You supplied {type(neighbors_for_ratio).__name__}."
        )

    if ratio_use <= 0:
        raise ValueError(f"neighbors_for_ratio resolved to a non-positive ratio ({ratio_use}).")

    assumed_number_of_possible_neighbors = int(np.ceil(ratio_use * np.ceil(np.sqrt(adata.shape[0]) / 2)))
    if assumed_number_of_possible_neighbors < 1:
        raise ValueError(
            f"neighbors_for_ratio implies {assumed_number_of_possible_neighbors} possible neighbors per spot. It must be at least 1."
        )
    if assumed_number_of_possible_neighbors >= adata.shape[0]:
        raise ValueError(
            f"neighbors_for_ratio implies {assumed_number_of_possible_neighbors} possible neighbors per spot, which must be less than the number of spots ({adata.shape[0]})."
        )

    # Checking if neighbors_for_ftu is actually a valid integer.
    if isinstance(neighbors_for_ftu, (bool, np.bool_)):
        raise TypeError("neighbors_for_ftu must be an integer.")
    elif isinstance(neighbors_for_ftu, Integral):
        neighbors_for_ftu = int(neighbors_for_ftu)
        if neighbors_for_ftu < 1:
            raise ValueError(f"neighbors_for_ftu must at least be 1. You supplied {neighbors_for_ftu!r}.")
        elif neighbors_for_ftu >= adata.shape[1]:
            raise ValueError(
                f"neighbors_for_ftu cannot be larger than the number of variants. You supplied {neighbors_for_ftu!r}. This is larger than or equal to the initial number of variants and the possible number of SVVs. Reduce it to a sensible number."
            )
    elif isinstance(neighbors_for_ftu, Real):
        value = float(neighbors_for_ftu)
        if not math.isfinite(value):
            raise ValueError(f"neighbors_for_ftu must be finite. You supplied {neighbors_for_ftu!r}.")
        neighbors_for_ftu = math.ceil(value)
        if verbose:
            print(f"neighbors_for_ftu is a float ({value}). ", flush=True)
            print(f"Rounding up to {neighbors_for_ftu}.", flush=True)
        if neighbors_for_ftu < 1:
            raise ValueError(f"neighbors_for_ftu must at least be 1. You supplied {neighbors_for_ftu!r}.")
        elif neighbors_for_ftu >= adata.shape[1]:
            raise ValueError(
                f"neighbors_for_ftu cannot be larger than the number of variants. You supplied {neighbors_for_ftu!r}. This is larger than or equal to the initial number of variants and the possible number of SVVs. Reduce it to a sensible number."
            )
    else:
        raise TypeError(f"neighbors_for_ftu must be int. You supplied {type(neighbors_for_ftu).__name__}.")

    if ftu_on_error not in ("warn", "raise"):
        raise ValueError(f"ftu_on_error must be warn or raise. You supplied {ftu_on_error!r}.")

    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError(f"normalize must be bool. You supplied a {type(normalize).__name__}.")

    if not isinstance(detect_svg_normalize_lap, (bool, np.bool_)):
        raise TypeError(
            f"detect_svg_normalize_lap must be bool. You supplied a {type(detect_svg_normalize_lap).__name__}."
        )

    if not isinstance(detect_svg_filter_peaks, (bool, np.bool_)):
        raise TypeError(
            f"detect_svg_filter_peaks must be bool. You supplied a {type(detect_svg_filter_peaks).__name__}."
        )

    if not isinstance(detect_svg_cal_pval, (bool, np.bool_)):
        raise TypeError(f"detect_svg_cal_pval must be bool. You supplied a {type(detect_svg_cal_pval).__name__}.")

    if not isinstance(perform_low_pass_enhancement, (bool, np.bool_)):
        raise TypeError(
            f"perform_low_pass_enhancement must be bool. You supplied a {type(perform_low_pass_enhancement).__name__}."
        )

    if not isinstance(perform_ftu, (bool, np.bool_)):
        raise TypeError(f"perform_ftu must be bool. You supplied a {type(perform_ftu).__name__}.")

    if not isinstance(ftu_weight_by_freq, (bool, np.bool_)):
        raise TypeError(f"ftu_weight_by_freq must be bool. You supplied a {type(ftu_weight_by_freq).__name__}.")

    if not isinstance(ftu_normalize_lap, (bool, np.bool_)):
        raise TypeError(f"ftu_normalize_lap must be bool. You supplied a {type(ftu_normalize_lap).__name__}.")

    if isinstance(detect_svg_S, (bool, np.bool_)):
        raise TypeError(f"detect_svg_S must be int or float. You supplied a {type(detect_svg_S).__name__}.")
    elif not isinstance(detect_svg_S, Real):
        raise TypeError(f"detect_svg_S must be int or float. You supplied a {type(detect_svg_S).__name__}.")
    elif detect_svg_S <= 0:
        raise ValueError(f"detect_svg_S must be larger than 0. You supplied {detect_svg_S!r}.")

    if isinstance(low_pass_enhancement_c, (bool, np.bool_)):
        raise TypeError(
            f"low_pass_enhancement_c must be a float. You supplied a {type(low_pass_enhancement_c).__name__}."
        )
    elif not isinstance(low_pass_enhancement_c, Real):
        raise TypeError(
            f"low_pass_enhancement_c must be a float. You supplied a {type(low_pass_enhancement_c).__name__}."
        )
    else:
        if not math.isfinite(low_pass_enhancement_c) or low_pass_enhancement_c < 0 or low_pass_enhancement_c >= 1:
            raise ValueError(
                f"low_pass_enhancement_c must be finite and 0 <= x < 1. You supplied {low_pass_enhancement_c!r}."
            )

    if isinstance(low_pass_enhancement_ratio_low_freq, str):
        if low_pass_enhancement_ratio_low_freq != "infer":
            raise ValueError(
                f"low_pass_enhancement_ratio_low_freq must be an integer, float, or infer. You supplied {low_pass_enhancement_ratio_low_freq!r}."
            )
    elif isinstance(low_pass_enhancement_ratio_low_freq, (bool, np.bool_)):
        raise TypeError(
            "low_pass_enhancement_ratio_low_freq must either be an integer, float or infer. You supplied a bool."
        )
    elif isinstance(low_pass_enhancement_ratio_low_freq, Integral):
        low_pass_enhancement_ratio_low_freq = float(low_pass_enhancement_ratio_low_freq)
    elif isinstance(low_pass_enhancement_ratio_low_freq, Real):
        low_pass_enhancement_ratio_low_freq = float(low_pass_enhancement_ratio_low_freq)
        if not math.isfinite(low_pass_enhancement_ratio_low_freq):
            raise ValueError(
                f"low_pass_enhancement_ratio_low_freq must be finite. You supplied {low_pass_enhancement_ratio_low_freq!r}."
            )
    else:
        raise TypeError(
            f"low_pass_enhancement_ratio_low_freq must either be an integer, float or infer. You supplied {type(low_pass_enhancement_ratio_low_freq).__name__}."
        )

    if not isinstance(low_pass_enhancement_ratio_low_freq, str):
        if low_pass_enhancement_ratio_low_freq <= 0:
            raise ValueError(
                f"low_pass_enhancement_ratio_low_freq must be > 0. You supplied {low_pass_enhancement_ratio_low_freq!r}."
            )

    if isinstance(ftu_random_state, (bool, np.bool_)):
        raise TypeError(f"ftu_random_state must be an integer. You supplied a {type(ftu_random_state).__name__}.")
    elif not isinstance(ftu_random_state, Integral):
        raise TypeError(f"ftu_random_state must be an integer. You supplied a {type(ftu_random_state).__name__}.")

    # Checking if ftu_resolution is valid.
    if isinstance(ftu_resolution, (bool, np.bool_)):
        raise TypeError("ftu_resolution cannot be a bool. Supply a float, list, or tuple.")
    elif isinstance(ftu_resolution, Real):
        if not math.isfinite(ftu_resolution) or ftu_resolution <= 0:
            raise ValueError(f"ftu_resolution must be finite and > 0. You supplied {ftu_resolution!r}.")
        # SpaGFT only accepts a Python float, int or list (np.float32 and np.int64 fail its assert).
        ftu_resolution = float(ftu_resolution)
    elif isinstance(ftu_resolution, tuple):
        if len(ftu_resolution) != 3:
            raise ValueError(
                f"When ftu_resolution is a tuple it must be (start, end, step). You supplied {len(ftu_resolution)} values."
            )
        start, end, step = ftu_resolution
        for name, value in (("start", start), ("end", end), ("step", step)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"ftu_resolution {name} must be a float. You supplied {type(value).__name__}.")
            if not math.isfinite(value):
                raise ValueError(f"ftu_resolution {name} must be a finite value. You supplied {value!r}.")
        if step <= 0:
            raise ValueError(f"ftu_resolution step must be > 0. You supplied {step!r}.")
        if start <= 0:
            raise ValueError(f"ftu_resolution start must be > 0. You supplied {start!r}.")
        if start >= end:
            raise ValueError(
                f"ftu_resolution start ({start}) must be smaller than end ({end}), otherwise no resolution is tested."
            )
        ftu_resolution = (float(start), float(end), float(step))
    elif isinstance(ftu_resolution, (list, np.ndarray)):
        values = list(ftu_resolution)
        if len(values) == 0:
            raise ValueError("ftu_resolution must not be empty.")
        for value in values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"Every ftu_resolution value must be a float. You supplied {type(value).__name__}.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Every ftu_resolution value must be finite value and > 0. You supplied {value!r}.")
        # A plain list of Python floats is what SpaGFT and sc.tl.louvain expect.
        ftu_resolution = [float(value) for value in values]
    else:
        raise TypeError(
            f"ftu_resolution must be a float, list, or tuple. You supplied {type(ftu_resolution).__name__}."
        )

    # End of checks
    ####################################################################################################
    if verbose:
        print("We reorder the spots.", flush=True)
    # First by the second coordinate column, then by the first.
    # numpy.lexsort treats the last key as the most important.
    # SpaGFT reads the coordinates straight from adata.obsm[spatial_key], so we do not
    # write them to obs. Writing them as array_row/array_col would overwrite Space Ranger's
    # real array indices with the (pixel) values from obsm.
    coords_order = np.lexsort((coords[:, 0], coords[:, 1]))
    adata = adata[coords_order, :].copy()
    adata.obsm[spatial_key] = coords[coords_order].astype(np.float64, copy=True)

    if filter_by_min_spots:
        if verbose:
            print(f"Subsetting the object to variants with at least {min_spots} spots.", flush=True)

        sc.pp.filter_genes(adata, min_cells=min_spots)

        if adata.n_vars == 0:
            raise ValueError("The filtering removed every variant. Aborting since no result is possible now.")

    if normalize:
        if verbose:
            print("Normalization of the data.", flush=True)

        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    if verbose:
        print("Determine the number of low-frequency FMs and high-frequency FMs.", flush=True)

    # determine_frequency_ratio caches its Fourier modes in adata.uns["FMs_after_select"], and
    # detect_svg reuses them whenever the mode counts match, which they always do here.
    # So the graph built here is the one detect_svg uses: it needs the same normalize_lap.
    (ratio_low, ratio_high) = spg.gft.determine_frequency_ratio(
        adata,
        ratio_neighbors=ratio_use,
        spatial_info=spatial_key,
        normalize_lap=detect_svg_normalize_lap,
    )

    if verbose:
        print("Detecting SVVs.", flush=True)

    variant_df = spg.detect_svg(
        adata,
        ratio_low_freq=ratio_low,
        ratio_high_freq=ratio_high,
        ratio_neighbors=ratio_use,
        spatial_info=spatial_key,  # SpaGFT accepts an obsm key directly
        filter_peaks=detect_svg_filter_peaks,
        S=detect_svg_S,
        cal_pval=detect_svg_cal_pval,
        normalize_lap=detect_svg_normalize_lap,
    )
    mask = variant_df.cutoff_gft_score.astype(bool)
    if detect_svg_cal_pval:
        mask &= variant_df.fdr < p_value_threshold_svvs

    svg_list = variant_df.index[mask].tolist()

    if not perform_low_pass_enhancement and not perform_ftu:
        return adata, variant_df, svg_list, None

    if perform_low_pass_enhancement:
        if verbose:
            print("Signal enhancement.", flush=True)

        adata.layers["X_original"] = adata.X.copy()
        adata = spg.low_pass_enhancement(
            adata,
            ratio_low_freq=low_pass_enhancement_ratio_low_freq,
            ratio_neighbors=ratio_use,
            c=low_pass_enhancement_c,
            spatial_info=spatial_key,  # SpaGFT accepts an obsm key directly
            normalize_lap=low_pass_enhancement_normalize_lap,
            inplace=True,
        )

    if not perform_ftu:
        return adata, variant_df, svg_list, None

    # Preparing the output object in case something fails.
    variant_df_enhanced = None

    if verbose:
        print("Grouping SVVs.", flush=True)

    if len(svg_list) == 0:
        if ftu_on_error == "raise":
            raise ValueError(
                "No SVVs were detected, therefore FTU identification cannot run. Consider relaxing p_value_threshold_svvs, detect_svg_S, or filter_by_min_spots."
            )
        warnings.warn(
            "No SVVs were detected, so FTU identification was skipped. Consider relaxing p_value_threshold_svvs, detect_svg_S, or filter_by_min_spots.",
            FTUWarning,
            stacklevel=2,
        )
        return adata, variant_df, svg_list, None

    # This depends on the data, not only on the parameters, so it follows ftu_on_error
    # like the empty svg_list above. Raising unconditionally would discard the finished
    # SVV detection and enhancement for a user who asked for warnings.
    if neighbors_for_ftu >= len(svg_list):
        if ftu_on_error == "raise":
            raise ValueError(
                f"neighbors_for_ftu ({neighbors_for_ftu}) must be less than the number of detected SVVs ({len(svg_list)})."
            )
        warnings.warn(
            f"neighbors_for_ftu ({neighbors_for_ftu}) is not less than the number of detected SVVs ({len(svg_list)}), so FTU identification was skipped. Reduce neighbors_for_ftu below {len(svg_list)}.",
            FTUWarning,
            stacklevel=2,
        )
        return adata, variant_df, svg_list, None

    # identify_ftu runs a PCA on the full object as a side effect (see _PCA_SIDE_EFFECT_KEYS).
    pca_snapshot = _snapshot_slots(adata, _PCA_SIDE_EFFECT_KEYS)

    if perform_low_pass_enhancement:
        if verbose:
            print("Grouping SVVs based on the enhanced signal.", flush=True)
        try:
            variant_df_enhanced, adata = spg.identify_ftu(
                adata,
                svg_list=svg_list,
                ratio_fms=ratio_low,
                ratio_neighbors=ratio_use,
                spatial_info=spatial_key,  # SpaGFT accepts an obsm key directly
                n_neighbors=neighbors_for_ftu,
                resolution=ftu_resolution,
                weight_by_freq=ftu_weight_by_freq,
                normalize_lap=ftu_normalize_lap,
                random_state=ftu_random_state,
            )
        except Exception as exc:
            if ftu_on_error == "raise":
                raise
            warnings.warn(
                f"FTU identification failed and was skipped for the enhanced data: {type(exc).__name__}: {exc}. The SVVs in svg_list are still valid though.",
                FTUWarning,
                stacklevel=2,
            )
            if verbose:
                traceback.print_exc()

            variant_df_enhanced = None
            _drop_ftu_results(adata)  # clear partial results that might clutter the object

        else:
            _move_ftu_results(adata, "_enhanced")

        finally:
            _restore_slots(adata, _PCA_SIDE_EFFECT_KEYS, pca_snapshot)

        # Setting the original values back since we now need them for the second FTU selection.
        adata.layers["X_enhanced"] = adata.X.copy()
        adata.X = adata.layers["X_original"].copy()

    if verbose:
        print("Grouping SVVs based on the unenhanced signal.", flush=True)

    try:
        variant_df, adata = spg.identify_ftu(
            adata,
            svg_list=svg_list,
            ratio_fms=ratio_low,
            ratio_neighbors=ratio_use,
            spatial_info=spatial_key,  # SpaGFT accepts an obsm key directly
            n_neighbors=neighbors_for_ftu,
            resolution=ftu_resolution,
            weight_by_freq=ftu_weight_by_freq,
            normalize_lap=ftu_normalize_lap,
            random_state=ftu_random_state,
        )
    except Exception as exc:
        if ftu_on_error == "raise":
            raise
        warnings.warn(
            f"FTU identification failed and was skipped: {type(exc).__name__}: {exc}. The SVVs in svg_list are still valid though.",
            FTUWarning,
            stacklevel=2,
        )
        if verbose:
            traceback.print_exc()

        _drop_ftu_results(adata)  # clear partial results that might clutter the object

    finally:
        _restore_slots(adata, _PCA_SIDE_EFFECT_KEYS, pca_snapshot)

    # If we enhanced the data, we set the results back to X.
    if perform_low_pass_enhancement:
        adata.X = adata.layers["X_enhanced"].copy()

    return adata, variant_df, svg_list, variant_df_enhanced


def _value_at_sorted_position(stored_values_sorted, negative_value_count, implicit_zero_count, sorted_position):
    """
    Value at a position of the virtually sorted column, without the implicit zeros.

    The sorted column is: stored negative values, then all implicit zeros, then the
    stored non-negative values (stored zeros sort among them, which is equivalent).
    """
    if sorted_position < negative_value_count:
        return stored_values_sorted[sorted_position]
    # end if statement

    if sorted_position < negative_value_count + implicit_zero_count:
        return 0.0
    # end if statement

    return stored_values_sorted[sorted_position - implicit_zero_count]


def _sparse_column_quantiles_type7(sparse_matrix_csc, observed_value_counts, probabilities):
    """
    Per-column quantiles of a CSC matrix.
    This is a port of R quantile(type = 7, na.rm = TRUE).

    Parameters
    ----------
    sparse_matrix_csc : scipy.sparse.csc_matrix
        Cells x variants. Stored entries must lie in observed cells only.
    observed_value_counts : numpy.ndarray
        Number of observed (non-NA in R) cells per column. Observed cells without a stored
        entry are implicit zeros and enter the quantile as 0.
    probabilities : sequence of float
        Quantile probabilities in [0, 1].

    Returns
    -------
    numpy.ndarray
        Shape (len(probabilities), number of columns). NaN for columns without observed cells,
        as R returns NA for an all-NA vector.
    """
    number_of_columns = sparse_matrix_csc.shape[1]
    quantile_values = np.full((len(probabilities), number_of_columns), np.nan, dtype=np.float64)
    column_pointers = sparse_matrix_csc.indptr
    all_stored_values = sparse_matrix_csc.data

    for column_index in range(number_of_columns):
        observed_value_count = int(observed_value_counts[column_index])
        if observed_value_count == 0:
            continue
        # end if statement

        stored_values_sorted = np.sort(
            all_stored_values[column_pointers[column_index] : column_pointers[column_index + 1]]
        )
        implicit_zero_count = observed_value_count - stored_values_sorted.size
        negative_value_count = int(np.searchsorted(stored_values_sorted, 0.0, side="left"))

        for probability_index, probability in enumerate(probabilities):
            # R, quantile.default, type 7 (written 0-based here):
            # index = (n - 1) * p, lo = floor(index), hi = ceiling(index),
            # Q = (1 - h) * x[lo] + h * x[hi] with h = index - lo.
            quantile_index = (observed_value_count - 1) * probability
            lower_position = math.floor(quantile_index)
            upper_position = math.ceil(quantile_index)
            lower_value = _value_at_sorted_position(
                stored_values_sorted, negative_value_count, implicit_zero_count, lower_position
            )
            upper_value = _value_at_sorted_position(
                stored_values_sorted, negative_value_count, implicit_zero_count, upper_position
            )
            interpolation_weight = quantile_index - lower_position
            if interpolation_weight > 0 and upper_value != lower_value:
                quantile_values[probability_index, column_index] = (
                    1 - interpolation_weight
                ) * lower_value + interpolation_weight * upper_value
            else:
                quantile_values[probability_index, column_index] = lower_value
            # end if statement
        # end for loop
    # end for loop

    return quantile_values


def Variant_Selection_Quantile(
    adata,
    *,
    min_coverage: int | float = 2,
    quantiles: Sequence[float] = (0.1, 0.9),
    thresholds: Sequence[float] = (0.1, 0.9),
    min_quality: int | float | None = None,
    mean_allele_frequency: int | float = 0,
    remove_nocall: bool = False,
    quality_column: str = "quality",
    verbose: bool = True,
) -> list[str]:
    """
    We get variants of interest using quantile thresholding.
    Adapted from Peter van Galen, https://github.com/petervangalen/MAESTER-2021.
    A variant is kept when its mean allele frequency is above `mean_allele_frequency`,
    its mean coverage is above `min_coverage`, the lower quantile of its allele
    frequencies is at most `thresholds[0]` and the upper quantile is at least
    `thresholds[1]`.

    The computation is sparse-native: quantiles are taken from the
    stored entries of each variant plus a count of implicit zeros,
    so no layer is densified.
    The input object is not modified.

    Parameters
    ----------
    adata : AnnData
        Cells x variants. Requires the layers fraction and coverage, plus the layer
        consensus when remove_nocall is True.
    min_coverage : int or float
        Minimum mean coverage per variant (strictly greater than).
    quantiles : sequence of two floats
        The lower and upper quantile probabilities, each in [0, 1]. Computed like R's
        quantile(type = 7).
    thresholds : sequence of two floats
        Allele frequency thresholds: lower quantile <= thresholds[0] and
        upper quantile >= thresholds[1].
    min_quality : int, float or None
        Minimum variant quality (strictly greater than), read from adata.var[quality_column].
        Variants with missing quality are removed. None disables the quality filter.
    mean_allele_frequency : int or float
        Minimum mean allele frequency (strictly greater than).
    remove_nocall : bool
        Should NoCall cells (consensus == 0) be disregarded? They are then excluded from the
        means and quantiles, like NA with na.rm = TRUE in R.
    quality_column : str
        Column in adata.var with the variant quality. The loaders write "quality";
        the R function uses rowData(SE)$VariantQuality.
    verbose : bool
        Should the function give updates?

    Returns
    -------
    list of str
        Names of the variants of interest, in the order of adata.var_names.
    """
    if not isinstance(verbose, (bool, np.bool_)):
        raise TypeError(f"verbose must be bool. You supplied a {type(verbose).__name__}.")

    if not isinstance(remove_nocall, (bool, np.bool_)):
        raise TypeError(f"remove_nocall must be bool. You supplied a {type(remove_nocall).__name__}.")

    if isinstance(min_coverage, (bool, np.bool_)) or not isinstance(min_coverage, Real):
        raise TypeError(f"min_coverage must be an integer or a float. You supplied a {type(min_coverage).__name__}.")
    elif not math.isfinite(min_coverage):
        raise ValueError(f"min_coverage must be finite. You supplied {min_coverage!r}.")

    if isinstance(mean_allele_frequency, (bool, np.bool_)) or not isinstance(mean_allele_frequency, Real):
        raise TypeError(
            f"mean_allele_frequency must be an integer or a float. You supplied a {type(mean_allele_frequency).__name__}."
        )
    elif not math.isfinite(mean_allele_frequency):
        raise ValueError(f"mean_allele_frequency must be finite. You supplied {mean_allele_frequency!r}.")

    if min_quality is not None:
        if isinstance(min_quality, (bool, np.bool_)) or not isinstance(min_quality, Real):
            raise TypeError(
                f"min_quality must be an integer, a float or None. You supplied a {type(min_quality).__name__}."
            )
        elif not math.isfinite(min_quality):
            raise ValueError(f"min_quality must be finite. You supplied {min_quality!r}.")
        # end if statement
    # end if statement

    for parameter_name, parameter_value in (("quantiles", quantiles), ("thresholds", thresholds)):
        if isinstance(parameter_value, (str, bytes)) or not isinstance(parameter_value, (Sequence, np.ndarray)):
            raise TypeError(
                f"{parameter_name} must be a sequence of two floats. You supplied a {type(parameter_value).__name__}."
            )
        elif len(parameter_value) != 2:
            raise ValueError(f"{parameter_name} must contain exactly two values. You supplied {len(parameter_value)}.")
        # end if statement

        for single_value in parameter_value:
            if isinstance(single_value, (bool, np.bool_)) or not isinstance(single_value, Real):
                raise TypeError(
                    f"Every value in {parameter_name} must be a float. You supplied a {type(single_value).__name__}."
                )
            elif not math.isfinite(single_value):
                raise ValueError(f"Every value in {parameter_name} must be finite. You supplied {single_value!r}.")
            # end if statement
        # end for loop
    # end for loop

    for probability in quantiles:
        if probability < 0 or probability > 1:
            raise ValueError(f"quantiles must lie in [0, 1]. You supplied {probability!r}.")
        # end if statement
    # end for loop

    if not isinstance(quality_column, str):
        raise TypeError(f"quality_column must be a string. You supplied a {type(quality_column).__name__}.")

    required_layers = ["fraction", "coverage"] + (["consensus"] if remove_nocall else [])
    missing_layers = [layer_name for layer_name in required_layers if layer_name not in adata.layers]
    if missing_layers:
        raise KeyError(f"adata.layers misses {missing_layers}. Available layers are: {list(adata.layers.keys())}.")

    if min_quality is not None and quality_column not in adata.var.columns:
        raise KeyError(
            f"min_quality is set, but adata.var has no column {quality_column!r}. "
            f"Available columns are: {list(adata.var.columns)}."
        )

    # End of checks
    ####################################################################################################
    # Float64 copies, so sums over ~100k cells do not accumulate in float32 and adata stays untouched.
    fraction_matrix = sp.csc_matrix(adata.layers["fraction"], dtype=np.float64, copy=True)
    coverage_matrix = sp.csc_matrix(adata.layers["coverage"], dtype=np.float64, copy=True)

    if np.isnan(fraction_matrix.data).any() or np.isnan(coverage_matrix.data).any():
        raise ValueError(
            "The fraction or coverage layer contains NaN. Missing values must be encoded via the "
            "consensus layer (remove_nocall = True), not as NaN in the layers."
        )

    if remove_nocall:
        if verbose:
            print("We set NoCall cells as NA.", flush=True)
        # end if statement

        # True where a consensus call exists (consensus == 0 means NoCall). Sparse, unlike == 0.
        called_mask = sp.csc_matrix(adata.layers["consensus"]) != 0
        fraction_matrix = sp.csc_matrix(fraction_matrix.multiply(called_mask))
        coverage_matrix = sp.csc_matrix(coverage_matrix.multiply(called_mask))
        # R: rowMeans/quantile with na.rm = TRUE divide by the non-NA cells, which are the
        # called cells, including reference calls whose allele frequency is exactly 0.
        observed_cells_per_variant = np.asarray(called_mask.sum(axis=0)).ravel().astype(np.int64)
    else:
        observed_cells_per_variant = np.full(adata.n_vars, adata.n_obs, dtype=np.int64)
    # end if statement

    fraction_matrix.sum_duplicates()

    if verbose:
        print("Get the mean allele frequency and coverage.", flush=True)
    # end if statement

    # axis = 0 reduces over cells, giving one value per variant (R: rowMeans over SE rows).
    fraction_sum_per_variant = np.asarray(fraction_matrix.sum(axis=0)).ravel()
    coverage_sum_per_variant = np.asarray(coverage_matrix.sum(axis=0)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_af = fraction_sum_per_variant / observed_cells_per_variant
        mean_coverage = coverage_sum_per_variant / observed_cells_per_variant
    # end with statement

    if verbose:
        print("Get the quantiles of the VAFs of each variant.", flush=True)
    # end if statement

    quantile_values = _sparse_column_quantiles_type7(
        fraction_matrix, observed_cells_per_variant, [float(probability) for probability in quantiles]
    )

    variant_statistics = pd.DataFrame(
        {
            "Mean_AF": mean_af,
            "Mean_Cov": mean_coverage,
            "Quantile1": quantile_values[0],
            "Quantile2": quantile_values[1],
        },
        index=adata.var_names,
    )

    keep_variant = np.ones(adata.n_vars, dtype=bool)
    if min_quality is not None:
        variant_quality = pd.to_numeric(adata.var[quality_column], errors="coerce").to_numpy(dtype=np.float64)
        keep_variant &= ~np.isnan(variant_quality) & (variant_quality > min_quality)
    # end if statement

    if verbose:
        print("Thresholding using the quantile approach.", flush=True)
    # end if statement

    # Comparisons with NaN are False, so variants without observed cells drop out, as in R's subset().
    keep_variant &= (
        (variant_statistics["Mean_AF"].to_numpy() > mean_allele_frequency)
        & (variant_statistics["Mean_Cov"].to_numpy() > min_coverage)
        & (variant_statistics["Quantile1"].to_numpy() <= thresholds[0])
        & (variant_statistics["Quantile2"].to_numpy() >= thresholds[1])
    )

    return variant_statistics.index[keep_variant].tolist()
