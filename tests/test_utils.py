# Tests for the functions in the _utils module. The erros in spyder should be
# okay for the most part. The helper functions are imported from conftest.py.

import warnings

import anndata as ad
import numpy as np
import scipy.sparse as sp

from spatial_sigurd_py._utils import _to_array

# Since _build_consensus is unused for now, we have to import it directly.
from spatial_sigurd_py._utils.basic import _build_consensus


###############################################################################
# Testing _to_array
###############################################################################
# Function to make an AnnData object. This will be used in the function
# test_sparse_layer_returns_transposed_dense_float32
def _make_adata(layer):
    # AnnData layers are (n_obs, n_vars) == (cells, variants)
    n_cells, n_variants = layer.shape
    return ad.AnnData(
        X=np.zeros((n_cells, n_variants), dtype=np.float32),
        layers={"counts": layer},
    )

class TestToArray:
    def test_sparse_layer_returns_transposed_dense_float32(self):
        dense = np.array([[1, 0], [0, 4], [5, 0]], dtype=np.float32)  # 3 cells x 2 variants
        adata = _make_adata(sp.csr_matrix(dense))
        
        out = _to_array("counts", adata)
        
        assert isinstance(out, np.ndarray)   # densified, not sparse
        assert not sp.issparse(out)
        assert out.dtype == np.float32        # default dtype
        assert out.shape == (2, 3)            # variants x cells (transposed)
        np.testing.assert_array_equal(out, dense.T)
    
    def test_dense_layer_returns_transposed_dense_float32(self):
        dense = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)  # 2 cells x 3 variants
        adata = _make_adata(dense)
        
        out = _to_array("counts", adata)
        
        assert out.dtype == np.float32        # cast down from float64
        assert out.shape == (3, 2)
        np.testing.assert_array_equal(out, dense.T.astype(np.float32))
    
    def test_custom_dtype_is_respected(self):
        adata = _make_adata(sp.csr_matrix(np.eye(2, dtype=np.float32)))
        assert _to_array("counts", adata, dtype=np.float64).dtype == np.float64
    
    def test_sparse_and_dense_paths_agree(self):
        dense = np.array([[0, 2], [3, 0], [0, 7]], dtype=np.float32)
        np.testing.assert_array_equal(
            _to_array("counts", _make_adata(dense)),
            _to_array("counts", _make_adata(sp.csr_matrix(dense))),
        )


###############################################################################
# Testing _build_consensus
###############################################################################
def _codes():
    # cov >= alt everywhere (coverage = alt + ref)
    # (0,0) cov=0 alt=0 -> 0 NoCall
    # (0,1) cov=5 alt=0 -> 1 Reference
    # (1,0) cov=5 alt=3 -> 2 Variant (partial alt)
    # (1,1) cov=8 alt=8 -> 2 Variant (all alt reads)
    cov = sp.csr_matrix(np.array([[0, 5], [5, 8]], dtype=np.int32))
    alt = sp.csr_matrix(np.array([[0, 0], [3, 8]], dtype=np.int32))
    return alt, cov

class TestBuildConsensus:
    def test_all_three_codes_and_variant_precedence(self):
        alt, cov = _codes()
        out = _build_consensus(alt, cov)
        np.testing.assert_array_equal(out.toarray(), np.array([[0, 1], [2, 2]]))
    
    def test_shape_and_orientation_preserved(self):
        alt, cov = _codes()
        out = _build_consensus(alt, cov)
        assert out.shape == cov.shape    # (n_cells, n_variants), no transpose
    
    def test_output_stays_sparse(self):
        alt, cov = _codes()
        assert sp.issparse(_build_consensus(alt, cov))
    
    def test_nocall_positions_not_stored(self):
        alt, cov = _codes()
        out = _build_consensus(alt, cov)
        assert out.nnz == 3               # the single 0 (NoCall) is dropped
        assert (out.data == 0).sum() == 0 # eliminate_zeros actually ran
    
    def test_default_dtype_is_int8(self):
        alt, cov = _codes()
        assert _build_consensus(alt, cov).dtype == np.int8
    
    def test_custom_dtype_respected(self):
        alt, cov = _codes()
        assert _build_consensus(alt, cov, dtype=np.int16).dtype == np.int16
    
    def test_stored_explicit_zero_in_alt_is_not_a_variant(self):
        # alt has a *stored* zero at (0,0); it must not be coded as a variant
        alt = sp.csr_matrix(
            (np.array([0, 3]), (np.array([0, 1]), np.array([0, 0]))), shape=(2, 1)
        )
        cov = sp.csr_matrix(np.array([[7], [7]], dtype=np.int32))
        assert _build_consensus(alt, cov).toarray()[0, 0] == 1   # Reference, not Variant
    
    def test_no_sparse_efficiency_warning(self):
        alt, cov = _codes()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _build_consensus(alt, cov)
