import gc
import numbers
from collections.abc import Collection

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from spatial_sigurd_py._utils import _to_array


def _estimate_grid_step(xy: np.ndarray) -> float:
    """Median nearest-neighbor distance (robust estimate of one grid step)."""
    tree = cKDTree(xy)
    dists, _ = tree.query(xy, k=2)  # [self, nearest-other]
    return float(np.median(dists[:, 1]))


def _build_adjacency_1step(xy: np.ndarray, radius_factor: float = 1.25) -> sp.csr_matrix:
    """
    Undirected 1-step adjacency matrix A (CSR, n_cells x n_cells).
    Spots within radius_factor * step are considered neighbors.
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
    # end for loop
    
    A = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)).tocsr()
    
    A = ((A + A.T) > 0).astype(np.int8).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def build_within2_neighborlists(xy: np.ndarray, radius_factor: float = 1.25) -> list[np.ndarray]:
    """
    Precompute, for each cell i, the cells within graph distance <= 2, excluding i.
    Returns a list of integer NumPy arrays.
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
        # end for loop
        
        scratch[i] = False
        idx = np.flatnonzero(scratch).astype(np.int32, copy=False)
        within2.append(idx)
        
        scratch[idx] = False
    # end for loop
    
    return within2


def _build_withink_neighborlists(xy: np.ndarray, k: int = 2, radius_factor: float = 1.25) -> list[np.ndarray]:
    """
    Precompute, for each cell i, the cells within graph distance <= k, excluding i.
    BFS to depth k. Returns a list of integer NumPy arrays.
    """
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, numbers.Integral) or k < 1:
        raise ValueError("k must be an integer >= 1.")
    # end if statement
    
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
                # end for loop
            # end for loop
            
            if not nxt:
                break
            # end if statement
            
            frontier = nxt
        # end for loop
        
        visited[i] = False
        idx = np.flatnonzero(visited).astype(np.int32, copy=False)
        withink.append(idx)
        visited[idx] = False
    # end for loop
    
    return withink


def _extract_xy_from_obsm(
    spatial,
    x_col=None,
    y_col=None,
) -> np.ndarray:
    """
    Accept either:
      - a pandas DataFrame in adata.obsm['spatial'] with named columns, or
      - a NumPy array / sparse matrix with at least 2 columns.
    
    If a DataFrame is given:
      - x_col / y_col can be column names or integer positions.
      - if omitted, the first two columns are used.
    
    If an array is given:
      - x_col / y_col can be integer positions.
      - if omitted, columns 0 and 1 are used.
    """
    # If x_col is bool/np.bool_ it would be registered as an integer. We check
    # for this and throw an error.
    if isinstance(x_col, (bool, np.bool_)):
        raise ValueError(f"x_col index {x_col} is a boolean value. Change it to an integer.")
    # end if statement
    
    # If x_col is not one of the correct types, we throw an error.
    if x_col is not None and not isinstance(x_col, (str, numbers.Integral)):
        raise TypeError("x_col must be None, a column name, or an integer column index.")
    # end if statement
    
    # If y_col is bool/np.bool_ it would be registered as an integer. We check
    # for this and throw an error.
    if isinstance(y_col, (bool, np.bool_)):
        raise ValueError(f"y_col index {y_col} is a boolean value. Change it to an integer.")
    # end if statement
    
    # If y_col is not one of the correct types, we throw an error.
    if y_col is not None and not isinstance(y_col, (str, numbers.Integral)):
        raise TypeError("y_col must be None, a column name, or an integer column index.")
    # end if statement
    
    # We check if spatial is a DataFrame, an array or a sparse matrix. If not,
    # we throw an error since we cannot deal with it.
    if not isinstance(spatial, (pd.DataFrame, np.ndarray)) and not sp.issparse(spatial):
        raise TypeError(f"spatial must be a DataFrame, NumPy array, or sparse matrix, got {type(spatial)}.")
    # end if statement
    
    # We need at least 2 dimensions. So, we check for at least two dimensions
    # and 2 columns specifically.
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] must have shape (n_cells, >=2).")
    # end if statement
    
    # We check if x_col is a int and that it actually refers to an existing
    # column.
    if isinstance(x_col, numbers.Integral):
        if x_col < 0 or x_col >= spatial.shape[1]:
            raise ValueError(f"x_col index {x_col} is out of bounds for spatial with {spatial.shape[1]} columns.")
        # end if statement
    # end if statement
    
    # We check if y_col is a int and that it actually refers to an existing
    # column.
    if isinstance(y_col, numbers.Integral):
        if y_col < 0 or y_col >= spatial.shape[1]:
            raise ValueError(f"y_col index {y_col} is out of bounds for spatial with {spatial.shape[1]} columns.")
        # end if statement
    # end if statement
    
    # If we are dealing with a DataFrame, we do this. This branch can terminate
    # without triggering the array branch below.
    if isinstance(spatial, pd.DataFrame):
        # If x_col is None, we simply take the first column.
        if x_col is None:
            x_col = spatial.columns[0]
        # end if statement
        
        # If y_col is None, we simply take the first column.
        if y_col is None:
            y_col = spatial.columns[1]
        # end if statement
        
        # We check if x_col is referring to an actual column.
        if isinstance(x_col, str) and x_col not in spatial.columns:
            raise ValueError(f"x_col '{x_col}' not found in spatial columns: {list(spatial.columns)}")
        # end if statement
        
        # We check if y_col is referring to an actual column.
        if isinstance(y_col, str) and y_col not in spatial.columns:
            raise ValueError(f"y_col '{y_col}' not found in spatial columns: {list(spatial.columns)}")
        # end if statement
        
        # We check if x_col is an integer. True: we get the column by index.
        # False: we get the column by name.
        if isinstance(x_col, numbers.Integral):
            x = spatial.iloc[:, x_col].to_numpy(dtype=np.float64, copy=False)
        else:
            x = spatial[x_col].to_numpy(dtype=np.float64, copy=False)
        # end if statement
        
        # We check if x_col is an integer. True: we get the column by index.
        # False: we get the column by name.
        if isinstance(y_col, numbers.Integral):
            y = spatial.iloc[:, y_col].to_numpy(dtype=np.float64, copy=False)
        else:
            y = spatial[y_col].to_numpy(dtype=np.float64, copy=False)
        # end if statement
        
        return np.column_stack([x, y])
    # end if statement
    
    # Array branch. We check that x/y_col are actually integers. Otherwise, we
    # cannot use them here. In the case of None, we just use the first/second
    # column. These will of course be integers since we defined them like it.
    xi = 0 if x_col is None else x_col
    yi = 1 if y_col is None else y_col
    
    # If x/y_col are not integers, we throw an error.
    if not isinstance(xi, numbers.Integral) or not isinstance(yi, numbers.Integral):
        raise TypeError("If adata.obsm['spatial'] is not a DataFrame, x_col and y_col must be integer column indices.")
    # end if statement
    
    # We get the columns from a sparse matrix.
    if sp.issparse(spatial):
        return np.column_stack(
            [
                spatial[:, xi].toarray().ravel(),
                spatial[:, yi].toarray().ravel(),
            ]
        ).astype(np.float64, copy=False)
    # end if statement
    
    # We get the the columns from a normal matrix.
    arr = np.asarray(spatial)
    return arr[:, [xi, yi]].astype(np.float64, copy=False)


def _find_rows_in_sorted_column(sorted_rows: np.ndarray, target_rows: np.ndarray):
    """
    Locate target_rows inside a sorted CSC/CSR index vector using searchsorted.
    Returns (positions, valid_mask).
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
        AnnData from one of the loading functions (**cells × variants** orientation).
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
    
    # alt_reads = adata.layers["alt_reads"].tolil(copy = True)
    # ref_reads = adata.layers["ref_reads"].tolil(copy = True)
    # fraction  = adata.layers["fraction"].tolil(copy = True)
    # coverage  = adata.layers["coverage"].tolil(copy = True)
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
    
    # 4. Removing variants based on minimum requirements.
    if verbose:
        print("Removing variants and cells that do not meet the required minimums.")
    
    # Persist modified layers back to AnnData (CSR keeps memory lower).
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
    
    alt_reads = adata.layers["alt_reads"]
    
    if verbose:
        print("Recalculating average coverage per variant.")
    
    adata.var["avg_coverage"] = np.asarray(adata.layers["coverage"].mean(axis=0)).ravel()
    
    if verbose:
        print("Recalculating strand concordance.")
    
    number_of_variants = adata.n_vars
    strand_concordance = np.empty(number_of_variants, dtype=np.float32)
    fw_reads = _to_array("alt_fwd", adata_object=adata)
    rev_reads = _to_array("alt_rev", adata_object=adata)
    for i in range(number_of_variants):
        fw = fw_reads[i]
        rv = rev_reads[i]
        mask = (fw > 0) | (rv > 0)  # ignore all-zero cells
        if mask.sum() < 2:
            strand_concordance[i] = np.nan
        else:
            strand_concordance[i] = np.corrcoef(fw[mask], rv[mask])[0, 1]
    
    adata.var["strand_concordance"] = strand_concordance
    
    gc.collect()
    if verbose:
        print(f"Finished filtering. {adata.n_obs} cells × {adata.n_vars} variants were retained.")
    
    return adata


def Pruning(
    adata: ad.AnnData,
    *,
    spatial_key: str = "spatial",
    x_col=None,
    y_col=None,
    min_alt_reads: int = 0,
    radius_factor: float = 1.25,
    k_hops: int = 2,
    min_pos_neighbors: int = 1,
    drop_zero_variants: bool = False,
    drop_zero_cells: bool = False,
    copy: bool = True,
    verbose: bool = True,
):
    """
    Remove isolated variant-positive spots/cells using precomputed <=2-hop neighbor lists.
    
    Isolation rule
    For each variant separately, a cell is considered isolated if:
      - alt_reads > min_alt_reads for that cell/variant, and
      - there is no other positive cell for the same variant within graph distance <= 2.
    
    Effects for each isolated cell/variant entry
      - alt_reads  -> set to 0
      - coverage   -> subtract the removed alt_reads
      - alt_fwd    -> set to 0
      - alt_rev    -> set to 0
      - fraction   -> set to 0
      - optionally drop cells / variants that become entirely zero in alt_reads
      - avg_coverage per variant is recalculated after optional dropping
    
    Notes
    -----
    - Assumes layers are stored as (cells x variants).
    - "Entirely zero" is defined from alt_reads / X, not from coverage.
    - ref_reads / ref_fwd / ref_rev are not altered except through optional subsetting.
    """
    # We are checking if all the layers we need are present. If any layer is missing
    # we stop and throw an error.
    required_layers = ["alt_reads", "coverage", "alt_fwd", "alt_rev", "fraction"]
    missing_layers = [k for k in required_layers if k not in adata.layers]
    if missing_layers:
        raise ValueError(f"adata is missing required layers: {missing_layers}")
    # end if statement

    # We check if the spatial information is present.
    if spatial_key not in adata.obsm:
        raise ValueError(f"adata.obsm['{spatial_key}'] not found.")
    # end if statement
    
    if isinstance(k_hops, (bool, np.bool_)) or not isinstance(k_hops, numbers.Integral) or k_hops < 1:
        raise ValueError("k_hops must be an integer >= 1.")
    # end if statement
    
    if (isinstance(min_pos_neighbors, (bool, np.bool_))
        or not isinstance(min_pos_neighbors, numbers.Integral)
        or min_pos_neighbors < 0
        ):
        raise ValueError("min_pos_neighbors must be an integer >= 0.")
    # end if statement
    
    out = adata.copy() if copy else adata
    
    # 1. Spatial coordinates and precomputed <=2-hop neighborhoods
    xy = _extract_xy_from_obsm(out.obsm[spatial_key], x_col=x_col, y_col=y_col)
    
    if xy.shape[0] != out.n_obs:
        raise ValueError(f"Spatial coordinates have {xy.shape[0]} rows, but adata has {out.n_obs} cells.")
    # end if statement
    
    if verbose:
        print(f"Precomputing <={k_hops}-hop neighbor lists.", flush=True)
    # end if statement
    
    khop_neighbors = _build_withink_neighborlists(xy, k=k_hops, radius_factor=radius_factor)
    
    # 2. Read layers as CSC for efficient per-variant operations
    if verbose:
        print("Loading layers.", flush=True)
    # end if statement
    
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
    # end for loop
    
    n_cells, n_vars = alt_reads.shape
    pruned_per_variant = np.zeros(n_vars, dtype=np.int32)
    
    # Reusable boolean mask for positive cells of one variant
    pos_mask = np.zeros(n_cells, dtype=bool)
    
    # 3. Prune isolated positives variant by variant
    if verbose:
        print("Pruning isolated spots.", flush=True)
    # end if statement
    
    for j in range(n_vars):
        a0, a1 = alt_reads.indptr[j], alt_reads.indptr[j + 1]
        alt_rows = alt_reads.indices[a0:a1]
        alt_data = alt_reads.data[a0:a1]
        
        expressed_mask = alt_data > min_alt_reads
        pos_rows = alt_rows[expressed_mask]
        
        if pos_rows.size == 0:
            continue
        # end if statement
        
        if pos_rows.size == 1:
            # A single positive spot has zero positive same-variant neighbors,
            # so it is isolated only when at least one is required.
            if min_pos_neighbors < 1:
                continue
            # end if statement
            
            iso_rows = pos_rows
            alt_local_pos = np.flatnonzero(expressed_mask)
        else:
            pos_mask[pos_rows] = True
            
            if min_pos_neighbors == 1:
                # Fast path: support needs only one positive neighbor (current default).
                supported = np.fromiter(
                    (pos_mask[khop_neighbors[i]].any() for i in pos_rows),
                    dtype=bool,
                    count=pos_rows.size,
                )
            else:
                counts = np.fromiter(
                    (int(pos_mask[khop_neighbors[i]].sum()) for i in pos_rows),
                    dtype=np.int64,
                    count=pos_rows.size,
                )
                supported = counts >= min_pos_neighbors
            # end if statement
            
            iso_mask = ~supported
            pos_mask[pos_rows] = False
            
            if not np.any(iso_mask):
                continue
            # end if statement
            
            iso_rows = pos_rows[iso_mask]
            expressed_positions = np.flatnonzero(expressed_mask)
            iso_positions_in_expr = np.flatnonzero(iso_mask)
            alt_local_pos = expressed_positions[iso_positions_in_expr]
        # end if statement
        
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
        # end if statement
        
        cov_data[cov_local_pos] -= removed_alt
        if np.any(cov_data[cov_local_pos] < 0):
            raise ValueError(f"Coverage became negative after pruning for variant {out.var_names[j]}.")
        # end if statement
        
        # alt_fwd -> 0
        f0, f1 = alt_fwd.indptr[j], alt_fwd.indptr[j + 1]
        fwd_rows = alt_fwd.indices[f0:f1]
        fwd_data = alt_fwd.data[f0:f1]
        
        fwd_local_pos, fwd_valid = _find_rows_in_sorted_column(fwd_rows, iso_rows)
        if np.any(fwd_valid):
            fwd_data[fwd_local_pos[fwd_valid]] = 0
        # end if statement
        
        # alt_rev -> 0
        r0, r1 = alt_rev.indptr[j], alt_rev.indptr[j + 1]
        rev_rows = alt_rev.indices[r0:r1]
        rev_data = alt_rev.data[r0:r1]
        
        rev_local_pos, rev_valid = _find_rows_in_sorted_column(rev_rows, iso_rows)
        if np.any(rev_valid):
            rev_data[rev_local_pos[rev_valid]] = 0
        # end if statement
        
        # fraction -> 0
        q0, q1 = fraction.indptr[j], fraction.indptr[j + 1]
        frac_rows = fraction.indices[q0:q1]
        frac_data = fraction.data[q0:q1]
        
        frac_local_pos, frac_valid = _find_rows_in_sorted_column(frac_rows, iso_rows)
        if np.any(frac_valid):
            frac_data[frac_local_pos[frac_valid]] = 0.0
        # end if statement
    # end for loop
    
    for M in (alt_reads, coverage, alt_fwd, alt_rev, fraction):
        M.eliminate_zeros()
    # end for loop
    
    # 4. Optionally drop zero cells / zero variants BEFORE avg_coverage
    if drop_zero_cells or drop_zero_variants:
        if verbose:
            print("Dropping zero cells/variants.", flush=True)
        # end if statement
        
        keep_cells = np.ones(alt_reads.shape[0], dtype=bool)
        keep_vars = np.ones(alt_reads.shape[1], dtype=bool)
        
        if drop_zero_cells:
            keep_cells = np.asarray(alt_reads.sum(axis=1)).ravel() > 0
        # end if statement
        
        if drop_zero_variants:
            keep_vars = np.asarray(alt_reads.sum(axis=0)).ravel() > 0
        # end if statement
        
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
            # end if statement
            
            if not np.all(keep_vars):
                out._inplace_subset_var(keep_vars)
            # end if statement
        # end if statement
    # end if statement
    
    # 5. Recalculate avg_coverage per variant AFTER optional dropping
    if verbose:
        print("Recalculating average coverage per variant.", flush=True)
    # end if statement
    
    if coverage.shape[1] == 0:
        avg_cov = np.array([], dtype=np.float64)
    elif coverage.shape[0] == 0:
        avg_cov = np.zeros(coverage.shape[1], dtype=np.float64)
    else:
        avg_cov = np.asarray(coverage.sum(axis=0)).ravel().astype(np.float64) / coverage.shape[0]
    # end if statement
    
    # 6. Write back
    out.X = alt_reads.copy()
    out.layers["alt_reads"] = alt_reads
    out.layers["coverage"] = coverage
    out.layers["alt_fwd"] = alt_fwd
    out.layers["alt_rev"] = alt_rev
    out.layers["fraction"] = fraction
    out.var["avg_coverage"] = avg_cov
    out.var["isolated_spots_pruned"] = pruned_per_variant
    
    if verbose:
        print(f"Finished pruning. Removed {int(pruned_per_variant.sum())} isolated cell-variant entries.", flush=True)
    # end if statement
    
    return out
