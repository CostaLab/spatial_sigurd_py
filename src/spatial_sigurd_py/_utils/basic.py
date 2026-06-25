import numpy as np
import scipy.sparse as sp


def _to_array(layer_name, adata_object, dtype=np.float32):
    """Layer ==> dense float32   (scipy sparse to ndarray)"""
    M = adata_object.layers[layer_name]
    if sp.issparse(M):
        return M.T.toarray().astype(dtype, copy=False)  # variants × cells
    # end if statement
    
    return np.asarray(M.T, dtype=dtype)


def _build_consensus(
    alt: sp.spmatrix,
    cov: sp.spmatrix,
    *,
    dtype: np.dtype = np.int8,
) -> sp.csr_matrix:
    """
    Create a consensus matrix:
    
        0 – NoCall   (coverage == 0)
        1 – Reference (coverage  > 0 & alt == 0)
        2 – Variant   (alt > 0)
    Shape = (n_cells, n_variants) – same orientation as all AnnData layers.
    """
    # For speed work in CSR (row major = cell-wise).
    alt_csr = alt.tocsr()
    cov_csr = cov.tocsr()
    
    # Variant mask ────────────────────────────────────────────────────────────
    variant_mask = alt_csr.copy()
    variant_mask.data = np.ones_like(variant_mask.data, dtype=dtype) * 2
    
    # Reference mask: (cov > 0) AND (alt == 0)
    ref_mask = (cov_csr > 0).astype(dtype)
    ref_mask[alt_csr.astype(bool)] = 0  # zero-out where we already have a variant
    ref_mask.data[:] = 1
    
    consensus = (variant_mask + ref_mask).astype(dtype, copy=False)
    consensus.eliminate_zeros()  # keep sparsity
    return consensus
