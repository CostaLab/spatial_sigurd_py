from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp


def _avg_coverage_per_variant(coverage):
    """Mean coverage per variant over all cells. coverage is cells x variants."""
    n_cells = coverage.shape[0]
    if coverage.shape[1] == 0:
        return np.array([], dtype = np.float64)
    
    if n_cells == 0:
        return np.zeros(coverage.shape[1], dtype = np.float64)
    
    return np.asarray(coverage.sum(axis = 0)).ravel().astype(np.float64) / n_cells


def _strand_concordance(alt_fwd, alt_rev) -> np.ndarray:
    """
    Per-variant Pearson correlation between forward and reverse alt reads, over
    informative cells only (fwd > 0 or rev > 0), matching the mgatk R reference.
    
    alt_fwd, alt_rev : cells x variants (native AnnData layer orientation),
                       sparse (any scipy format) or dense.
    Returns          : float32, length n_vars. NaN where < 2 informative cells or
                       where either strand has zero variance across them.
    """
    A = alt_fwd.tocsc() if sp.issparse(alt_fwd) else sp.csc_matrix(np.asarray(alt_fwd))
    B = alt_rev.tocsc() if sp.issparse(alt_rev) else sp.csc_matrix(np.asarray(alt_rev))
    
    if A.shape != B.shape:
        raise ValueError(f"alt_fwd {A.shape} and alt_rev {B.shape} must have the same shape.")
    
    n_cells, n_vars = A.shape
    
    if n_vars == 0:
        return np.array([], dtype = np.float32)
    
    if n_cells == 0:
        return np.full(n_vars, np.nan, dtype = np.float32)
    
    A = A.astype(np.float64, copy=False)
    B = B.astype(np.float64, copy=False)
    
    sx  = np.asarray(A.sum(axis = 0)).ravel()
    sy  = np.asarray(B.sum(axis = 0)).ravel()
    sxx = np.asarray(A.multiply(A).sum(axis = 0)).ravel()
    syy = np.asarray(B.multiply(B).sum(axis = 0)).ravel()
    sxy = np.asarray(A.multiply(B).sum(axis = 0)).ravel()
    
    n_inf = _informative_cells_per_variant(A, B)
    
    with np.errstate(invalid = "ignore", divide = "ignore"):
        ex = sx / n_inf
        ey = sy / n_inf
        vx = np.clip(sxx / n_inf - ex * ex, 0.0, None)
        vy = np.clip(syy / n_inf - ey * ey, 0.0, None)
        cov = sxy / n_inf - ex * ey
        
        denom = np.sqrt(vx * vy)
        corr = cov / denom
    
    valid = (n_inf >= 2) & (denom > 0)
    return np.where(valid, corr, np.nan).astype(np.float32, copy=False)


def _recompute_variant_stats(adata):
    """
    Single source of truth for the derived per-variant .var columns. Every function
    that mutates alt_fwd / alt_rev / coverage must call this so the stored values
    always match the current layers.
    """
    adata.var["avg_coverage"] = _avg_coverage_per_variant(adata.layers["coverage"])
    adata.var["strand_concordance"] = _strand_concordance(
        adata.layers["alt_fwd"],
        adata.layers["alt_rev"],
    )
    return adata


def _to_array(layer_name, adata_object, dtype = np.float32):
    """Layer ==> dense float32   (scipy sparse to ndarray)"""
    M = adata_object.layers[layer_name]
    if sp.issparse(M):
        return M.T.toarray().astype(dtype, copy=False) # variants × cells
    
    return np.asarray(M.T, dtype=dtype)


def _build_consensus(
    alt: sp.spmatrix,
    cov: sp.spmatrix,
    *,
    dtype: np.dtype = np.int8,
) -> sp.csr_matrix:
    """
    Create a consensus matrix with the following values:
    
        0 – NoCall    (coverage == 0)
        1 – Reference (coverage > 0 & alt == 0)
        2 – Variant   (alt > 0)
    Shape = (n_cells, n_variants)
    """
    # To improve speed, we work with in CSR matrices (row major = cell-wise).
    alt_csr = alt.tocsr()
    cov_csr = cov.tocsr()
    
    # The logic: first we convert to a boolean matrix.
    # If cov > 0, we have at least a reference.
    # If alts > 0, we have alt. This does not contradict cov, since cov is alts + refs.
    reference = (cov_csr > 0).astype(dtype)   # 1 wherever a position is covered
    variant = (alt_csr > 0).astype(dtype) * 2 # 2 wherever alternative reads are observed
    
    # maximum() gives variant precedence: 2 > 1 > 0
    consensus = reference.maximum(variant).astype(dtype, copy=False)
    consensus.eliminate_zeros()
    
    return consensus


def _safe_div_sparse(
    num: sp.spmatrix,
    den: sp.spmatrix,
    out_dtype=np.float32,
    *,
    check: bool = True,
) -> sp.csc_matrix:
    """
    Element-wise numerator (num) / denominator (den) for sparse matrices,
    returning 0 where den == 0.
    
    The output carries nonzeros only where BOTH num and den are nonzero, so any
    num entry sitting over a zero denominator is silently collapsed to 0 rather
    than producing inf. That collapse is only correct if num has no nonzeros
    outside den's support; with check=True this invariant is verified and a
    ValueError is raised otherwise.
    
    """
    if num.shape != den.shape:
        raise ValueError(f"num {num.shape} and den {den.shape} must have the same shape.")
    
    den = den.tocsc()
    
    if check:
        # num nonzero AND den zero. Uses den's *value* support, so an explicitly
        # stored zero in den counts as zero and is caught here too.
        offending = num.astype(bool) > den.astype(bool)
        if offending.nnz:
            raise ValueError(
                f"{offending.nnz} entries have a nonzero numerator over a zero denominator;"
                "the quotient there is undefined, not 0. Fix the inputs or pass check=False."
            )
    
    inv = den.astype(np.float32, copy=True)
    inv.eliminate_zeros() # Here, we drop explicitly-stored zeros so 1/0 can't sneak in.
    inv.data = 1.0 / inv.data
    return num.astype(np.float32, copy=False).multiply(inv).tocsc().astype(out_dtype, copy=False)


def read_visium_parquet_map(
    path: str | Path,
    *,
    bam_col: str = "square_002um",
    cell_col: str = "cell_id",
    in_cell_col: str = "in_cell",
    cells_include: set[str] | None = None,
    cells_exclude: set[str] | None = None,
) -> pl.DataFrame:
    """
    Reads Visium-HD parquet mapping and returns a Polars DataFrame with:
    cell (spatial barcode) and cell_id (final cell barcode)
    Filtered to in_cell==True and optional include/exclude on cell_id.
    """
    m = (
        pl.scan_parquet(str(path))
        .select([bam_col, cell_col, in_cell_col])
        .filter(pl.col(in_cell_col))
        .select([pl.col(bam_col).alias("cell"), pl.col(cell_col).alias("cell_id")])
        .filter(pl.col("cell").is_not_null() & pl.col("cell_id").is_not_null())
        .with_columns(pl.col("cell_id").cast(pl.Utf8))
        .collect()
    )
    if cells_include is not None:
        m = m.filter(pl.col("cell_id").is_in(list(cells_include)))
    
    if cells_exclude is not None:
        m = m.filter(~pl.col("cell_id").is_in(list(cells_exclude)))
    
    return m.unique(subset=["cell", "cell_id"])


def _informative_cells_per_variant(alt_fwd: sp.csc_matrix, alt_rev: sp.csc_matrix) -> np.ndarray:
    """
    Number of informative cells per variant, i.e. cells where at least one strand
    carries an alt read: |(fwd > 0) | (rev > 0)|. Cells x variants in, one count
    per variant out.
    """
    support = (alt_fwd != 0).maximum(alt_rev != 0) # logical OR of the two patterns
    return support.getnnz(axis = 0).astype(np.float64, copy=False)
