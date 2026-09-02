import gc
import numbers
from collections.abc import Collection
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from spatial_sigurd_py._utils.basic import _recompute_variant_stats


def _estimate_grid_step(xy: np.ndarray) -> float:
    """
    Median nearest-neighbor distance (robust estimate of one grid step).

    This function gets the median distance between the spots.
    We can then use this for the downstream analysis to prune isolated spots.

    Parameters
    ----------
    xy: the positions as a numpy array

    Returns
    -------
    a float variable for the median distance

    """
    tree = cKDTree(xy)
    dists, _ = tree.query(xy, k=2)  # [self, nearest-other]
    return float(np.median(dists[:, 1]))


def _build_adjacency_1step(xy: np.ndarray, radius_factor: float = 1.25) -> sp.csr_matrix:
    """
    Undirected 1-step adjacency matrix A (CSR, n_cells x n_cells).
    Spots within radius_factor * step are considered neighbors.

    Parameters
    ----------
    xy: positions as a numpy array
    radius_factor: float variable that enlarges or shrinks the median distance.
                   The median distance is estimated by _estimate_grid_step.

    Returns
    -------
    A: a sparse coo_matrix with the neighbors.

    """
    n = xy.shape[0]
    step = _estimate_grid_step(xy)
    r = radius_factor * step

    tree = cKDTree(xy)
    neigh_lists = tree.query_ball_point(xy, r=r)

    rows, cols = [], []
    for i, neigh in enumerate(neigh_lists):
        for j in neigh:
            if j != i:
                rows.append(i)
                cols.append(j)

    A = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)).tocsr()

    A = ((A + A.T) > 0).astype(np.int8).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def build_within2_neighborlists(xy: np.ndarray, radius_factor: float = 1.25) -> list[np.ndarray]:
    """
    Precompute, for each cell i, the cells within graph distance <= 2, excluding i.
    This is a specific version of build_withink_neighborlists.
    build_withink_neighborlists does the same, but for a specified k, instead of 2.
    Returns a list of integer numpy arrays.

    Parameters
    ----------
    xy: positions as numpy array.
    radius_factor: float that enlarges or shrinks the distance estimated
                   by _estimate_grid_step via _build_adjacency_1step.

    Returns
    -------
    within2: list of integer numpy arrays that contain the neighbors.

    """
    A = _build_adjacency_1step(xy, radius_factor=radius_factor).tocsr()
    n = A.shape[0]

    nbr1 = [A.indices[A.indptr[i] : A.indptr[i + 1]] for i in range(n)]
    within2 = []
    scratch = np.zeros(n, dtype=bool)

    for i in range(n):
        # 1-hop
        scratch[nbr1[i]] = True

        # 2-hop
        for j in nbr1[i]:
            scratch[nbr1[j]] = True

        scratch[i] = False
        idx = np.flatnonzero(scratch).astype(np.int32, copy=False)
        within2.append(idx)

        scratch[idx] = False

    return within2


def build_withink_neighborlists(xy: np.ndarray, k: int = 2, radius_factor: float = 1.25) -> list[np.ndarray]:
    """
    Precompute, for each cell i, the cells within graph distance <= k, excluding i.
    This is a specific version of build_within2_neighborlists.
    build_withink_neighborlists does the same, but for a specified k, instead of k.
    Returns a list of integer numpy arrays.

    Parameters
    ----------
    xy: positions as numpy array.
    k: integer variable that indicates the number of hops.
    radius_factor: float that enlarges or shrinks the distance estimated
                   by _estimate_grid_step via _build_adjacency_1step.

    Returns
    -------
    withink: list of integer numpy arrays that contain the neighbors.
    """
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, numbers.Integral) or k < 1:
        raise ValueError("k must be an integer >= 1.")

    A = _build_adjacency_1step(xy, radius_factor=radius_factor).tocsr()
    n = A.shape[0]
    nbr1 = [A.indices[A.indptr[i] : A.indptr[i + 1]] for i in range(n)]

    withink = []
    visited = np.zeros(n, dtype=bool)

    for i in range(n):
        visited[i] = True
        frontier = [i]
        for _ in range(k):
            nxt = []
            for u in frontier:
                for w in nbr1[u]:
                    if not visited[w]:
                        visited[w] = True
                        nxt.append(w)

            if not nxt:
                break

            frontier = nxt

        visited[i] = False
        idx = np.flatnonzero(visited).astype(np.int32, copy=False)
        withink.append(idx)
        visited[idx] = False

    return withink


def _extract_xy_from_obsm(
    spatial,
    x_col=None,
    y_col=None,
) -> np.ndarray:
    """
    Extracts the positions from an anndata object and returns it as an array.

    The function accepts either:
      - a pandas DataFrame in adata.obsm['spatial'] with named columns, or
      - a numpy array / sparse matrix with at least 2 columns.

    If a DataFrame is provided:
      - x_col / y_col can be column names or integer positions, or
      - if omitted, the first two columns are used.

    If an array is provided:
      - x_col / y_col can be integer positions, or
      - if omitted, columns 0 and 1 are used.

    Parameters
    ----------
    spatial: the input object as described above.
    x_col: name of the column containing the x coordinates
    y_col: name of the column containing the y coordinates

    Returns
    -------
    numpy array that contains the positions.
    """
    # If x_col is bool/np.bool_ it would be registered as an integer. We check
    # for this and throw an error.
    if isinstance(x_col, (bool, np.bool_)):
        raise ValueError(f"x_col index {x_col} is a boolean value. Change it to an integer.")

    # If x_col is not one of the correct types, we throw an error.
    if x_col is not None and not isinstance(x_col, (str, numbers.Integral)):
        raise TypeError("x_col must be None, a column name, or an integer column index.")

    # If y_col is bool/np.bool_ it would be registered as an integer. We check
    # for this and throw an error.
    if isinstance(y_col, (bool, np.bool_)):
        raise ValueError(f"y_col index {y_col} is a boolean value. Change it to an integer.")

    # If y_col is not one of the correct types, we throw an error.
    if y_col is not None and not isinstance(y_col, (str, numbers.Integral)):
        raise TypeError("y_col must be None, a column name, or an integer column index.")

    # We check if spatial is a DataFrame, an array or a sparse matrix. If not,
    # we throw an error since we cannot deal with it.
    if not isinstance(spatial, (pd.DataFrame, np.ndarray)) and not sp.issparse(spatial):
        raise TypeError(f"spatial must be a DataFrame, NumPy array, or sparse matrix, got {type(spatial)}.")

    # We need at least 2 dimensions. So, we check for at least two dimensions
    # and 2 columns specifically.
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] must have shape (n_cells, >=2).")

    # We check if x_col is a int and that it actually refers to an existing
    # column.
    if isinstance(x_col, numbers.Integral):
        if x_col < 0 or x_col >= spatial.shape[1]:
            raise ValueError(f"x_col index {x_col} is out of bounds for spatial with {spatial.shape[1]} columns.")

    # We check if y_col is a int and that it actually refers to an existing
    # column.
    if isinstance(y_col, numbers.Integral):
        if y_col < 0 or y_col >= spatial.shape[1]:
            raise ValueError(f"y_col index {y_col} is out of bounds for spatial with {spatial.shape[1]} columns.")

    # If we are dealing with a DataFrame, we do this. This branch can terminate
    # without triggering the array branch below.
    if isinstance(spatial, pd.DataFrame):
        # If x_col is None, we simply take the first column.
        if x_col is None:
            x_col = spatial.columns[0]

        # If y_col is None, we simply take the first column.
        if y_col is None:
            y_col = spatial.columns[1]

        # We check if x_col is referring to an actual column.
        if isinstance(x_col, str) and x_col not in spatial.columns:
            raise ValueError(f"x_col '{x_col}' not found in spatial columns: {list(spatial.columns)}")

        # We check if y_col is referring to an actual column.
        if isinstance(y_col, str) and y_col not in spatial.columns:
            raise ValueError(f"y_col '{y_col}' not found in spatial columns: {list(spatial.columns)}")

        # We check if x_col is an integer. True: we get the column by index.
        # False: we get the column by name.
        if isinstance(x_col, numbers.Integral):
            x = spatial.iloc[:, x_col].to_numpy(dtype=np.float64, copy=False)
        else:
            x = spatial[x_col].to_numpy(dtype=np.float64, copy=False)

        # We check if x_col is an integer. True: we get the column by index.
        # False: we get the column by name.
        if isinstance(y_col, numbers.Integral):
            y = spatial.iloc[:, y_col].to_numpy(dtype=np.float64, copy=False)
        else:
            y = spatial[y_col].to_numpy(dtype=np.float64, copy=False)

        return np.column_stack([x, y])

    # Array branch. We check that x/y_col are actually integers. Otherwise, we
    # cannot use them here. In the case of None, we just use the first/second
    # column. These will of course be integers since we defined them like it.
    xi = 0 if x_col is None else x_col
    yi = 1 if y_col is None else y_col

    # If x/y_col are not integers, we throw an error.
    if not isinstance(xi, numbers.Integral) or not isinstance(yi, numbers.Integral):
        raise TypeError("If adata.obsm['spatial'] is not a DataFrame, x_col and y_col must be integer column indices.")

    # We get the columns from a sparse matrix.
    if sp.issparse(spatial):
        return np.column_stack(
            [
                spatial[:, xi].toarray().ravel(),
                spatial[:, yi].toarray().ravel(),
            ]
        ).astype(np.float64, copy=False)

    # We get the the columns from a normal matrix.
    arr = np.asarray(spatial)
    return arr[:, [xi, yi]].astype(np.float64, copy=False)


def _find_rows_in_sorted_column(sorted_rows: np.ndarray, target_rows: np.ndarray):
    """
    Locate target_rows inside a sorted CSC/CSR index vector using searchsorted.
    We do this to ensure that we have no missing variants.
    Returns (positions, valid_mask).

    Parameters
    ----------
    sorted_rows: numpy array like the cov_rows in the Pruning function.
    target_rows: numpy array like iso_rows in the Pruning function.

    Returns
    -------
    pos: the positions that are present
    valid: mask of all valid positions. This should be all True.

    """
    pos = np.searchsorted(sorted_rows, target_rows)
    valid = pos < sorted_rows.size
    valid[valid] = sorted_rows[pos[valid]] == target_rows[valid]
    return pos, valid


def _mask_from_coords(rows: np.ndarray, cols: np.ndarray, n_rows: int, n_cols: int) -> sp.coo_matrix:
    # Binary mask at (rows, cols); COO works well for multiply
    data = np.ones_like(rows, dtype=np.float32)
    return sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


def _zero_at_mask(mat: sp.csr_matrix, M: sp.coo_matrix) -> sp.csr_matrix:
    # Set mat[mask] = 0 by mat - mat∘M
    out = mat - mat.multiply(M)
    out.eliminate_zeros()
    return out


def _subtract_other_at_mask(dst: sp.csr_matrix, src: sp.csr_matrix, M: sp.coo_matrix) -> sp.csr_matrix:
    # dst[mask] -= src[mask]
    sub = src.multiply(M)
    out = dst - sub
    out.eliminate_zeros()
    return out


def Filtering(
    adata: ad.AnnData,
    cells_include: Collection[str] | None = None,
    cells_exclude: Collection[str] | None = None,
    fraction_threshold: float | None = None,
    keep_refs: bool = True,
    alts_threshold: int | None = None,
    min_cells_per_variant: int | None = None,
    min_variants_per_cell: int | None = None,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Filtering of an annData object containing variant information.

    Parameters
    ----------
    adata
        AnnData from one of the loading functions (**cells x variants** orientation).
        Must contain the layers:
        `alt_reads, ref_reads, fraction, coverage.
        Optional layers: alt_fwd, alt_rev, ref_fwd, ref_rev are handled if present.
    cells_include, cells_exclude
        Keep / remove these barcodes (uses adata.obs_names).
    fraction_threshold
        If 0 < VAF < threshold the variant is removed (set to 0).
    keep_refs
        When VAF < threshold, keep reference reads?
    alts_threshold
        If alt_reads < threshold the variant is removed (set to 0).
    min_cells_per_variant
        Drop variants not detected in at least this many cells.
    min_variants_per_cell
        Drop cells with < this many variants/refs covered.
    verbose
        Should the function comment on what it is doing?

    Result
    ------
    adata: the filtered anndata object
    """
    if verbose:
        print("Performing sanity checks.")

    required_layers = ["alt_reads", "ref_reads", "fraction", "coverage"]
    for layer in required_layers:
        if layer not in adata.layers:
            raise KeyError(f"Required layer '{layer}' not in AnnData object.")

    if not isinstance(keep_refs, bool):
        raise TypeError("keep_refs must be True or False.")

    if (fraction_threshold is not None) and ((fraction_threshold <= 0) or (fraction_threshold >= 1)):
        raise ValueError("fraction_threshold must be 0 < x < 1.")

    if (alts_threshold is not None) and ((not np.isfinite(alts_threshold)) or (alts_threshold < 0)):
        raise ValueError("alts_threshold must be a positive, finite number.")

    if (min_cells_per_variant is not None) and (
        (not np.isfinite(min_cells_per_variant)) or (min_cells_per_variant < 0)
    ):
        raise ValueError("min_cells_per_variant must be a positive, finite number.")

    if (min_cells_per_variant is not None) and min_cells_per_variant > adata.n_obs:
        raise ValueError("min_cells_per_variant is larger than the number of cells in the object.")

    if (min_variants_per_cell is not None) and (
        (not np.isfinite(min_variants_per_cell)) or (min_variants_per_cell < 0)
    ):
        raise ValueError("min_variants_per_cell must be a positive, finite number.")

    if (min_variants_per_cell is not None) and min_variants_per_cell > adata.n_vars:
        raise ValueError("min_variants_per_cell is larger than the number of variants in the object.")

    # 1. Removing cells based on lists.
    if cells_include is not None:
        if verbose:
            print("Keeping only cells in cells_include.")

        mask = adata.obs_names.isin(cells_include)
        if not mask.any():
            raise ValueError("None of the requested cells are present.")

        adata = adata[mask, :]

    if cells_exclude is not None:
        if verbose:
            print("Removing cells in cells_exclude.")

        mask = ~adata.obs_names.isin(cells_exclude)
        if not mask.any():
            raise ValueError("All cells would be removed by cells_exclude.")

        adata = adata[mask, :]

    # 2. Extracting and converting matrices.
    # At this point we convert the Compressed Sparse Row (CSR) matrices to List in List (LiL) sparse matrices.
    # Reason: A LiL matrix is a list of lists in which each row is a list in the object.
    # If we want to access many different rows and columns, we can very quickly do this in the LiL format.
    # The CSR format would need to build many internal arrays, which is much slower.
    # The CSR format should be much better for arithmetic. So, we convert back to CSR after we are done with the
    # processing.
    if verbose:
        print("Extracting layers after subsetting.")

    alt_reads = adata.layers["alt_reads"].tocsr()
    ref_reads = adata.layers["ref_reads"].tocsr()
    fraction = adata.layers["fraction"].tocsr()
    coverage = adata.layers["coverage"].tocsr()

    # Checking if optional layers are present.
    optional_layers = ["alt_fwd", "alt_rev", "ref_fwd", "ref_rev"]
    optional_data = {}

    for layer_use in optional_layers:
        if layer_use in adata.layers:
            optional_data[layer_use] = adata.layers[layer_use].tocsr()

    n_rows, n_cols = alt_reads.shape

    # 3. Fraction and alt reads thresholding.
    if fraction_threshold is not None:
        if verbose:
            print(f"Applying the fraction_threshold of {fraction_threshold}.")

        frac_coo = fraction.tocoo()
        selected = (frac_coo.data > 0) & (frac_coo.data < float(fraction_threshold))

        if selected.size:
            rows, cols = frac_coo.row[selected], frac_coo.col[selected]
            M = _mask_from_coords(rows, cols, n_rows, n_cols)

            # fraction -> 0 at mask
            fraction = _zero_at_mask(fraction, M)

            # coverage -= alt_reads at mask (use alt values BEFORE zeroing them)
            coverage = _subtract_other_at_mask(coverage, alt_reads, M)

            # alt reads -> 0 at mask
            alt_reads = _zero_at_mask(alt_reads, M)

            # strand-specific ALT (if present) -> 0 at mask
            if "alt_fwd" in optional_data:
                optional_data["alt_fwd"] = _zero_at_mask(optional_data["alt_fwd"], M)

            if "alt_rev" in optional_data:
                optional_data["alt_rev"] = _zero_at_mask(optional_data["alt_rev"], M)

            if not keep_refs:
                if verbose:
                    print("Removing reference reads after alt_reads thresholding.")

                if "ref_fwd" in optional_data:
                    optional_data["ref_fwd"] = _zero_at_mask(optional_data["ref_fwd"], M)

                if "ref_rev" in optional_data:
                    optional_data["ref_rev"] = _zero_at_mask(optional_data["ref_rev"], M)

                ref_reads = _zero_at_mask(ref_reads, M)
                coverage = _zero_at_mask(coverage, M)

        del frac_coo
        del rows
        del cols

    if alts_threshold is not None:
        if verbose:
            print(f"Applying alts_threshold = {alts_threshold}.")

        alt_coo = alt_reads.tocoo()
        selected = alt_coo.data < int(alts_threshold)

        if selected.any():
            rows, cols = alt_coo.row[selected], alt_coo.col[selected]
            M = _mask_from_coords(rows, cols, n_rows, n_cols)

            # fraction -> 0 at mask
            fraction = _zero_at_mask(fraction, M)

            # coverage -= alt_reads at mask (use current alt values before zeroing)
            coverage = _subtract_other_at_mask(coverage, alt_reads, M)

            # alt reads -> 0 at mask
            alt_reads = _zero_at_mask(alt_reads, M)

            # strand-specific ALT (if present) -> 0 at mask
            if "alt_fwd" in optional_data:
                optional_data["alt_fwd"] = _zero_at_mask(optional_data["alt_fwd"], M)

            if "alt_rev" in optional_data:
                optional_data["alt_rev"] = _zero_at_mask(optional_data["alt_rev"], M)

            # Optional: remove REF too
            if not keep_refs:
                if verbose:
                    print("Removing reference reads after alt_reads thresholding.")

                if "ref_fwd" in optional_data:
                    optional_data["ref_fwd"] = _zero_at_mask(optional_data["ref_fwd"], M)

                if "ref_rev" in optional_data:
                    optional_data["ref_rev"] = _zero_at_mask(optional_data["ref_rev"], M)

                ref_reads = _zero_at_mask(ref_reads, M)
                coverage = _zero_at_mask(coverage, M)

    # 4. Removing variants based on the minimum requirements.
    if verbose:
        print("Removing variants and cells that do not meet the required minimums.")

    # Persist modified layers back to AnnData (since CSR keeps memory lower).
    adata.layers["alt_reads"] = alt_reads
    adata.layers["ref_reads"] = ref_reads
    adata.layers["coverage"] = coverage
    adata.layers["fraction"] = fraction
    adata.X = alt_reads

    for k, v in optional_data.items():
        adata.layers[k] = v

    del optional_data

    if verbose:
        print("Removing variants without any alt or ref reads.")

    alt_nonzero = (alt_reads > 0).sum(axis=0).A1 > 0
    ref_nonzero = (ref_reads > 0).sum(axis=0).A1 > 0
    keep_variants = alt_nonzero | ref_nonzero
    del ref_reads

    if not keep_variants.any():
        raise ValueError("All variants would be removed after filtering.")

    adata = adata[:, keep_variants]

    if min_cells_per_variant is not None:
        if verbose:
            print(f"Filtering variants present in fewer than {min_cells_per_variant} cells.")

        alt_reads = adata.layers["alt_reads"]
        variant_present = (alt_reads > 0).sum(axis=0).A1
        keep_variants = variant_present >= min_cells_per_variant
        adata = adata[:, keep_variants]

    if min_variants_per_cell is not None:
        if verbose:
            print(f"Filtering cells with fewer than {min_variants_per_cell} variants.")

        alt_reads = adata.layers["alt_reads"]
        cell_present = (alt_reads > 0).sum(axis=1).A1
        keep_cells = cell_present >= min_variants_per_cell
        adata = adata[keep_cells, :]

    if verbose:
        print("Recalculating average coverage and strand concordance per variant.")

    _recompute_variant_stats(adata)

    gc.collect()
    if verbose:
        print(f"Finished filtering. {adata.n_obs} cells x {adata.n_vars} variants were retained.")

    return adata


def build_within_radius_neighborlists(xy: np.ndarray, radius: float) -> list[np.ndarray]:
    """
    Precompute, for each cell i, all cells within Euclidean distance <= radius,
    excluding i. Distance is measured in the coordinate units of `xy`, so convert
    to micrometers beforehand if you want `radius` to mean microns.

    Unlike build_withink_neighborlists, this builds no grid graph and estimates no
    grid step. It is meant for irregularly / randomly placed cells where a fixed
    physical search radius is more meaningful than a k-hop graph distance.
    """
    if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, numbers.Real) or radius <= 0:
        raise ValueError("radius must be a real number > 0.")

    tree = cKDTree(xy)
    neigh_lists = tree.query_ball_point(xy, r=radius, return_sorted=True)

    within_r = []
    for i, lst in enumerate(neigh_lists):
        arr = np.asarray(lst, dtype=np.int32)
        arr = arr[arr != i]  # exclude self
        within_r.append(arr)

    return within_r


def _scale_coords_to_um(xy: np.ndarray, microns_per_pixel=None) -> np.ndarray:
    """
    Convert coordinates from pixels to micrometers.

    If microns_per_pixel is None the coordinates are assumed to already be in the
    unit `radius_um` is measured in and are returned unchanged. Otherwise every
    coordinate is multiplied by microns_per_pixel.
    """
    if microns_per_pixel is None:
        return xy

    if (
        isinstance(microns_per_pixel, (bool, np.bool_))
        or not isinstance(microns_per_pixel, numbers.Real)
        or microns_per_pixel <= 0
    ):
        raise ValueError("microns_per_pixel must be a real number > 0 (or None).")

    return xy * float(microns_per_pixel)


def build_knn_neighborlists(xy: np.ndarray, k: int, max_distance=None) -> list[np.ndarray]:
    """
    For each cell i, its k nearest neighbors (excluding i), optionally capped at a
    maximum Euclidean distance. Distance is measured in the coordinate units of xy,
    so convert to micrometers beforehand if you want max_distance to mean microns.

    Unlike the radius builder, the neighborhood size is fixed at k regardless of how
    far the neighbors sit (unless max_distance trims it). In sparse regions KNN will
    therefore reach arbitrarily far; max_distance is the guard against that.
    """
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, numbers.Integral) or k < 1:
        raise ValueError("k must be an integer >= 1.")

    if max_distance is not None:
        if (
            isinstance(max_distance, (bool, np.bool_))
            or not isinstance(max_distance, numbers.Real)
            or max_distance <= 0
        ):
            raise ValueError("max_distance must be a real number > 0 (or None).")

    n = xy.shape[0]
    tree = cKDTree(xy)

    # Query k+1 because the nearest point to i is i itself (distance 0).
    k_query = min(k + 1, n)
    dists, idxs = tree.query(xy, k=k_query)

    # query returns 1D arrays when k_query == 1; force 2D.
    if dists.ndim == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    knn = []
    for i in range(n):
        row_idx = idxs[i]
        row_dst = dists[i]

        # Drop self and any padding (scipy pads with index == n and dist == inf
        # when fewer than k_query points exist).
        keep = (row_idx != i) & (row_idx < n) & np.isfinite(row_dst)

        if max_distance is not None:
            keep &= row_dst <= max_distance

        # query returns neighbors in ascending distance, so [:k] keeps the closest k.
        neigh = row_idx[keep][:k].astype(np.int32, copy=False)
        knn.append(neigh)

    return knn


def Pruning(
    adata: ad.AnnData,
    *,
    spatial_key: str = "spatial",
    x_col=None,
    y_col=None,
    min_alt_reads: int = 0,
    neighbor_mode: Literal["hops", "radius", "knn"] = "hops",
    radius_factor: float = 1.25,
    k_hops: int = 2,
    radius_um: float | None = None,
    n_neighbors: int | None = None,
    knn_max_distance_um: float | None = None,
    microns_per_pixel: float | None = None,
    min_pos_neighbors: int = 1,
    drop_zero_variants: bool = False,
    drop_zero_cells: bool = False,
    copy: bool = True,
    verbose: bool = True,
):
    """
    Remove isolated variant-positive spots/cells using precomputed neighbor lists.

    Isolation rule
    For each variant separately, a cell is considered isolated if there is
    no other positive cell for the same variant within given distance or
    neighborhood.
    Three different different methods implemented are:
        - K-Hops
        Spots/cells within a 2 hop distance are considered when determining
        isolation. Example: one to the right, one to the top is in the
        'neighborhood' of the original cell. This is a sensible distance for a
        strict latice as in spatial ATAC seq.
        - Radius
        This is a euclidean distance approach. If a spot is within the radius
        of circle centered on the spot of interest, it is in the neighborhood.
        This should be used if the size of the relevant structure is already
        known. Please note that pixel distances can be converted to actual
        distances using the scaling factor found in the spaceranger output.
        - KNN
        Instead of using a predetermined distance, the K nearest neighbors are
        selected and used as the neighborhood. This is useful in all cases,
        but is intended for spatial data that is not on a defined latice. In
        this case, the distance between cells might differ extremely and a KNN
        neighborhood ensures that we always have a neighborhood of this size.

    Effects for each isolated spot/variant entry:
        - alt_reads -> set to 0
        - coverage  -> subtract the removed alt_reads
        - alt_fwd   -> set to 0
        - alt_rev   -> set to 0
        - fraction  -> set to 0
        - optionally drop cells / variants that become entirely zero in alt_reads
        - avg_coverage per variant is recalculated after optional dropping

    Parameters
    ----------
    - radius_use: The radius used to determine the neighborhood. This is supposed
                  to be supplied in microns, but can be supplied in any distance
                  desired. If pixel distances are used, you can convert them before
                  to microns or supply the conversion ratio using microns_per_pixel.
    - radius_factor: A scaling factor for the K-hops pruning approach. Only used
                     if neighbor_mode = 'hops'. Not use for the radius approach.
                     If this increases the distance use over sqrt(2) * distance,
                     this turns the approach effectively into the Chebyshev distance.
                     Then, the distance is larger than the diagonal on the lattice.
                     Default = 1.25
    - microns_per_pixel: The microns per pixel ratio from the spaceranger results.

    Return
    ------
    out: the pruned anndata object

    Notes
    -----
    - Assumes layers are stored as (cells x variants).
    - "Entirely zero" is defined from alt_reads / X, not from coverage.
      A spot with reads, but without any variants, is probably not interesting
      for the analysis.
    - ref_reads / ref_fwd / ref_rev are not altered except through optional subsetting.
      Since we do not consider them for the pruning, we will not alter them.
    """
    if neighbor_mode not in ("hops", "radius", "knn"):
        raise ValueError("neighbor_mode must be one of 'hops', 'radius', or 'knn'.")

    if (
        isinstance(min_pos_neighbors, (bool, np.bool_))
        or not isinstance(min_pos_neighbors, numbers.Integral)
        or min_pos_neighbors < 0
    ):
        raise ValueError("min_pos_neighbors must be an integer >= 0.")

    if neighbor_mode == "hops":
        if isinstance(k_hops, (bool, np.bool_)) or not isinstance(k_hops, numbers.Integral) or k_hops < 1:
            raise ValueError("k_hops must be an integer >= 1.")
    elif neighbor_mode == "radius":
        if radius_um is None:
            raise ValueError("neighbor_mode='radius' requires radius_um to be set.")

        if isinstance(radius_um, (bool, np.bool_)) or not isinstance(radius_um, numbers.Real) or radius_um <= 0:
            raise ValueError("radius_um must be a real number > 0.")
    else:  # knn
        if n_neighbors is None:
            raise ValueError("neighbor_mode='knn' requires n_neighbors to be set.")

        if (
            isinstance(n_neighbors, (bool, np.bool_))
            or not isinstance(n_neighbors, numbers.Integral)
            or n_neighbors < 1
        ):
            raise ValueError("n_neighbors must be an integer >= 1.")

        if knn_max_distance_um is not None:
            if (
                isinstance(knn_max_distance_um, (bool, np.bool_))
                or not isinstance(knn_max_distance_um, numbers.Real)
                or knn_max_distance_um <= 0
            ):
                raise ValueError("knn_max_distance_um must be a real number > 0 (or None).")

    # We are checking if all the layers we need are present. If any layer is missing
    # we stop and throw an error.
    required_layers = ["alt_reads", "coverage", "alt_fwd", "alt_rev", "fraction"]
    missing_layers = [k for k in required_layers if k not in adata.layers]
    if missing_layers:
        raise ValueError(f"adata is missing required layers: {missing_layers}")

    # We check if the spatial information is present.
    if spatial_key not in adata.obsm:
        raise ValueError(f"adata.obsm['{spatial_key}'] not found.")

    out = adata.copy() if copy else adata

    # 1. Spatial coordinates and precomputed <=2-hop neighborhoods
    xy = _extract_xy_from_obsm(out.obsm[spatial_key], x_col=x_col, y_col=y_col)

    if xy.shape[0] != out.n_obs:
        raise ValueError(f"Spatial coordinates have {xy.shape[0]} rows, but adata has {out.n_obs} cells.")

    # Convert pixels -> micrometers if a pixel size was given. Scale-invariant in
    # hop mode; only actually changes behaviour in radius / knn mode.
    xy = _scale_coords_to_um(xy, microns_per_pixel=microns_per_pixel)
    unit = "um" if microns_per_pixel is not None else "coordinate units"

    if neighbor_mode == "hops":
        if verbose:
            print(f"Precomputing <={k_hops}-hop neighbor lists.", flush=True)
        neighbors = build_withink_neighborlists(xy, k=k_hops, radius_factor=radius_factor)
    elif neighbor_mode == "radius":
        if verbose:
            print(f"Precomputing neighbor lists within radius {radius_um} {unit}.", flush=True)
        neighbors = build_within_radius_neighborlists(xy, radius=radius_um)
    else:  # knn
        if verbose:
            cap = "" if knn_max_distance_um is None else f" (capped at {knn_max_distance_um} {unit})"
            print(f"Precomputing {n_neighbors}-nearest-neighbor lists{cap}.", flush=True)
        neighbors = build_knn_neighborlists(xy, k=n_neighbors, max_distance=knn_max_distance_um)

    # 2. Read layers as CSC for efficient per-variant operations
    if verbose:
        print("Loading layers.", flush=True)

    alt_reads = out.layers["alt_reads"]
    coverage = out.layers["coverage"]
    alt_fwd = out.layers["alt_fwd"]
    alt_rev = out.layers["alt_rev"]
    fraction = out.layers["fraction"]

    alt_reads = alt_reads.tocsc(copy=True) if sp.issparse(alt_reads) else sp.csc_matrix(alt_reads)
    coverage = coverage.tocsc(copy=True) if sp.issparse(coverage) else sp.csc_matrix(coverage)
    alt_fwd = alt_fwd.tocsc(copy=True) if sp.issparse(alt_fwd) else sp.csc_matrix(alt_fwd)
    alt_rev = alt_rev.tocsc(copy=True) if sp.issparse(alt_rev) else sp.csc_matrix(alt_rev)
    fraction = fraction.tocsc(copy=True) if sp.issparse(fraction) else sp.csc_matrix(fraction)

    for M in (alt_reads, coverage, alt_fwd, alt_rev, fraction):
        M.sort_indices()

    n_cells, n_vars = alt_reads.shape
    pruned_per_variant = np.zeros(n_vars, dtype=np.int32)

    # Reusable boolean mask for positive cells of one variant
    pos_mask = np.zeros(n_cells, dtype=bool)

    # 3. Prune isolated positives variant by variant
    if verbose:
        print("Pruning isolated spots.", flush=True)

    for j in range(n_vars):
        a0, a1 = alt_reads.indptr[j], alt_reads.indptr[j + 1]
        alt_rows = alt_reads.indices[a0:a1]
        alt_data = alt_reads.data[a0:a1]

        expressed_mask = alt_data >= min_alt_reads
        pos_rows = alt_rows[expressed_mask]

        if pos_rows.size == 0:
            continue

        if pos_rows.size == 1:
            # A single positive spot has zero positive same-variant neighbors,
            # so it is isolated only when at least one is required.
            if min_pos_neighbors < 1:
                continue

            iso_rows = pos_rows
            alt_local_pos = np.flatnonzero(expressed_mask)
        else:
            pos_mask[pos_rows] = True

            if min_pos_neighbors == 1:
                # Fast path: support needs only one positive neighbor (current default).
                supported = np.fromiter(
                    (pos_mask[neighbors[i]].any() for i in pos_rows),
                    dtype=bool,
                    count=pos_rows.size,
                )
            else:
                counts = np.fromiter(
                    (int(pos_mask[neighbors[i]].sum()) for i in pos_rows),
                    dtype=np.int64,
                    count=pos_rows.size,
                )
                supported = counts >= min_pos_neighbors

            iso_mask = ~supported
            pos_mask[pos_rows] = False

            if not np.any(iso_mask):
                continue

            iso_rows = pos_rows[iso_mask]
            expressed_positions = np.flatnonzero(expressed_mask)
            iso_positions_in_expr = np.flatnonzero(iso_mask)
            alt_local_pos = expressed_positions[iso_positions_in_expr]

        pruned_per_variant[j] = iso_rows.size
        removed_alt = alt_data[alt_local_pos].copy()

        # alt_reads -> 0
        alt_data[alt_local_pos] = 0

        # coverage -> subtract removed alt reads
        c0, c1 = coverage.indptr[j], coverage.indptr[j + 1]
        cov_rows = coverage.indices[c0:c1]
        cov_data = coverage.data[c0:c1]

        cov_local_pos, cov_valid = _find_rows_in_sorted_column(cov_rows, iso_rows)
        if not np.all(cov_valid):
            raise ValueError(
                f"Coverage layer is missing entries for some nonzero alt_reads in variant {out.var_names[j]}."
            )

        cov_data[cov_local_pos] -= removed_alt
        if np.any(cov_data[cov_local_pos] < 0):
            raise ValueError(f"Coverage became negative after pruning for variant {out.var_names[j]}.")

        # alt_fwd -> 0
        f0, f1 = alt_fwd.indptr[j], alt_fwd.indptr[j + 1]
        fwd_rows = alt_fwd.indices[f0:f1]
        fwd_data = alt_fwd.data[f0:f1]

        fwd_local_pos, fwd_valid = _find_rows_in_sorted_column(fwd_rows, iso_rows)
        if np.any(fwd_valid):
            fwd_data[fwd_local_pos[fwd_valid]] = 0

        # alt_rev -> 0
        r0, r1 = alt_rev.indptr[j], alt_rev.indptr[j + 1]
        rev_rows = alt_rev.indices[r0:r1]
        rev_data = alt_rev.data[r0:r1]

        rev_local_pos, rev_valid = _find_rows_in_sorted_column(rev_rows, iso_rows)
        if np.any(rev_valid):
            rev_data[rev_local_pos[rev_valid]] = 0

        # fraction -> 0
        q0, q1 = fraction.indptr[j], fraction.indptr[j + 1]
        frac_rows = fraction.indices[q0:q1]
        frac_data = fraction.data[q0:q1]

        frac_local_pos, frac_valid = _find_rows_in_sorted_column(frac_rows, iso_rows)
        if np.any(frac_valid):
            frac_data[frac_local_pos[frac_valid]] = 0.0

    for M in (alt_reads, coverage, alt_fwd, alt_rev, fraction):
        M.eliminate_zeros()

    # 4. Optionally drop zero cells / zero variants BEFORE avg_coverage
    if drop_zero_cells or drop_zero_variants:
        if verbose:
            print("Dropping zero cells/variants.", flush=True)

        keep_cells = np.ones(alt_reads.shape[0], dtype=bool)
        keep_vars = np.ones(alt_reads.shape[1], dtype=bool)

        if drop_zero_cells:
            keep_cells = np.asarray(alt_reads.sum(axis=1)).ravel() > 0

        if drop_zero_variants:
            keep_vars = np.asarray(alt_reads.sum(axis=0)).ravel() > 0

        alt_reads = alt_reads[keep_cells, :][:, keep_vars].tocsc()
        coverage = coverage[keep_cells, :][:, keep_vars].tocsc()
        alt_fwd = alt_fwd[keep_cells, :][:, keep_vars].tocsc()
        alt_rev = alt_rev[keep_cells, :][:, keep_vars].tocsc()
        fraction = fraction[keep_cells, :][:, keep_vars].tocsc()
        pruned_per_variant = pruned_per_variant[keep_vars]

        if copy:
            out = out[keep_cells, keep_vars].copy()
        else:
            if not np.all(keep_cells):
                out._inplace_subset_obs(keep_cells)

            if not np.all(keep_vars):
                out._inplace_subset_var(keep_vars)

    # 5. Write back the updated layers
    out.X = alt_reads.copy()
    out.layers["alt_reads"] = alt_reads
    out.layers["coverage"] = coverage
    out.layers["alt_fwd"] = alt_fwd
    out.layers["alt_rev"] = alt_rev
    out.layers["fraction"] = fraction

    # 6. Refresh ALL derived per-variant statistics from the updated layers
    if verbose:
        print("Recalculating average coverage and strand concordance per variant.", flush=True)
    _recompute_variant_stats(out)
    out.var["isolated_spots_pruned"] = pruned_per_variant

    if verbose:
        print(f"Finished pruning. Removed {int(pruned_per_variant.sum())} isolated cell-variant entries.", flush=True)

    return out
