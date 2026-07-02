# Tests for the functions in the _utils module.

import anndata as ad
import numpy as np
import scipy.sparse as sp

from spatial_sigurd_py._utils import _to_array


# Function to make an AnnData object. This will be used in the function
# test_sparse_layer_returns_transposed_dense_float32
def _make_adata(layer):
    # AnnData layers are (n_obs, n_vars) == (cells, variants)
    n_cells, n_variants = layer.shape
    return ad.AnnData(
        X = np.zeros((n_cells, n_variants), dtype = np.float32),
        layers = {"counts": layer},
    )


def test_sparse_layer_returns_transposed_dense_float32():
    dense = np.array([[1, 0], [0, 4], [5, 0]], dtype = np.float32)  # 3 cells x 2 variants
    adata = _make_adata(sp.csr_matrix(dense))
    
    out = _to_array("counts", adata)
    
    assert isinstance(out, np.ndarray)   # densified, not sparse
    assert not sp.issparse(out)
    assert out.dtype == np.float32        # default dtype
    assert out.shape == (2, 3)            # variants x cells (transposed)
    np.testing.assert_array_equal(out, dense.T)


def test_dense_layer_returns_transposed_dense_float32():
    dense = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)  # 2 cells x 3 variants
    adata = _make_adata(dense)
    
    out = _to_array("counts", adata)
    
    assert out.dtype == np.float32        # cast down from float64
    assert out.shape == (3, 2)
    np.testing.assert_array_equal(out, dense.T.astype(np.float32))


def test_custom_dtype_is_respected():
    adata = _make_adata(sp.csr_matrix(np.eye(2, dtype=np.float32)))
    assert _to_array("counts", adata, dtype=np.float64).dtype == np.float64


def test_sparse_and_dense_paths_agree():
    dense = np.array([[0, 2], [3, 0], [0, 7]], dtype=np.float32)
    np.testing.assert_array_equal(
        _to_array("counts", _make_adata(dense)),
        _to_array("counts", _make_adata(sp.csr_matrix(dense))),
    )
