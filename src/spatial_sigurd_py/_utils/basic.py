import numpy as np
import scipy.sparse as sp


def _to_array(layer_name, adata_object, dtype = np.float32):
    """Layer ==> dense float32   (scipy sparse to ndarray)"""
    M = adata_object.layers[layer_name]
    if sp.issparse(M):
        return M.T.toarray().astype(dtype, copy = False)  # variants × cells
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
    
        0 – NoCall    (coverage == 0)
        1 – Reference (coverage > 0 & alt == 0)
        2 – Variant   (alt > 0)
    Shape = (n_cells, n_variants) – same orientation as all AnnData layers.
    """
    # For speed work in CSR (row major = cell-wise).
    alt_csr = alt.tocsr()
    cov_csr = cov.tocsr()
    
    # The logic: first we convert to a boolean matrix.
    # If cov > 0, we have at least a reference.
    # If alts > 0, we have alt. This does not contradict cov, since cov is alts + refs.
    reference = (cov_csr > 0).astype(dtype)   # 1 wherever a position is covered
    variant = (alt_csr > 0).astype(dtype) * 2 # 2 wherever alternative reads are observed
    
    # maximum() gives variant precedence: 2 > 1 > 0
    consensus = reference.maximum(variant).astype(dtype, copy = False)
    consensus.eliminate_zeros()
    
    return consensus
