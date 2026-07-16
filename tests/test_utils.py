# Tests for the functions in the _utils module. The erros in spyder should be
# okay for the most part. The helper functions are imported from conftest.py.

import warnings

import anndata as ad
import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from spatial_sigurd_py._utils import (
    _avg_coverage_per_variant,
    _recompute_variant_stats,
    _safe_div_sparse,
    _strand_concordance,
    _to_array,
    read_visium_parquet_map,
)

# _build_consensus and _informative_cells_per_variant
# are not re-exported from the package __init__, so import them from .basic directly.
from spatial_sigurd_py._utils.basic import (
    _build_consensus,
    _informative_cells_per_variant,
)

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
# Constructors used to run the same test across dense / CSR / CSC inputs.
_FORMATS = {
    "dense": lambda d: np.asarray(d),
    "csr": lambda d: sp.csr_matrix(d),
    "csc": lambda d: sp.csc_matrix(d),
}


def _make_adata(layer):
    # AnnData layers are (n_obs, n_vars) == (cells, variants)
    n_cells, n_variants = layer.shape
    return ad.AnnData(
        X=np.zeros((n_cells, n_variants), dtype=np.float32),
        layers={"counts": layer},
    )


def _make_variant_adata(fwd, rev, cov):
    """AnnData carrying the three layers _recompute_variant_stats reads from."""
    n_cells, n_vars = fwd.shape
    return ad.AnnData(
        X=np.zeros((n_cells, n_vars), dtype=np.float32),
        layers={
            "alt_fwd": sp.csr_matrix(fwd),
            "alt_rev": sp.csr_matrix(rev),
            "coverage": sp.csr_matrix(cov),
        },
    )


def _concordance_oracle(fwd, rev):
    """
    Independent reference for _strand_concordance: masked np.corrcoef, per variant,
    over informative cells (fwd > 0 or rev > 0) only. NaN when < 2 informative
    cells or when either strand has zero variance across them.
    """
    fwd = np.asarray(fwd, dtype=np.float64)
    rev = np.asarray(rev, dtype=np.float64)
    out = []
    for j in range(fwd.shape[1]):
        f, r = fwd[:, j], rev[:, j]
        mask = (f > 0) | (r > 0)
        f, r = f[mask], r[mask]
        if f.size < 2 or f.std() == 0 or r.std() == 0:
            out.append(np.nan)
        else:
            out.append(np.corrcoef(f, r)[0, 1])
    
    return np.array(out, dtype=np.float32)


###############################################################################
# Testing _to_array
###############################################################################
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


###############################################################################
# Testing _avg_coverage_per_variant
###############################################################################
class TestAvgCoveragePerVariant:
    # coverage is cells x variants; result is per-variant column mean over ALL cells.
    @pytest.mark.parametrize("fmt", list(_FORMATS))
    def test_known_means(self, fmt):
        dense = np.array([[0, 4], [2, 6], [4, 8]], dtype=np.float64)  # 3 cells x 2 vars
        out = _avg_coverage_per_variant(_FORMATS[fmt](dense))
        np.testing.assert_allclose(out, [2.0, 6.0])   # colsums [6,18] / 3
    
    def test_dtype_is_float64(self):
        out = _avg_coverage_per_variant(sp.csr_matrix(np.array([[1, 2], [3, 4]])))
        assert out.dtype == np.float64
    
    def test_output_is_1d_length_nvars(self):
        out = _avg_coverage_per_variant(sp.csr_matrix(np.ones((5, 3))))
        assert out.shape == (3,)
    
    def test_empty_variants_returns_empty_float64(self):
        out = _avg_coverage_per_variant(sp.csr_matrix((4, 0)))
        assert out.shape == (0,)
        assert out.dtype == np.float64
    
    def test_zero_cells_returns_zeros(self):
        # no cells but 3 variants -> zeros, not a divide-by-zero
        out = _avg_coverage_per_variant(sp.csr_matrix((0, 3)))
        np.testing.assert_array_equal(out, np.zeros(3))
        assert out.dtype == np.float64
    
    def test_zero_by_zero_returns_empty(self):
        assert _avg_coverage_per_variant(sp.csr_matrix((0, 0))).shape == (0,)
    
    def test_sparse_and_dense_agree(self):
        dense = np.array([[0, 5, 1], [2, 0, 3], [4, 1, 0]], dtype=np.float64)
        np.testing.assert_allclose(
            _avg_coverage_per_variant(dense),
            _avg_coverage_per_variant(sp.csc_matrix(dense)),
        )
    
    def test_stored_explicit_zeros_do_not_change_sum(self):
        data = np.array([0.0, 3.0])            # a stored 0 at (0,0)
        m = sp.csc_matrix((data, (np.array([0, 1]), np.array([0, 0]))), shape=(2, 1))
        np.testing.assert_allclose(_avg_coverage_per_variant(m), [1.5])  # (0+3)/2


###############################################################################
# Testing _informative_cells_per_variant
###############################################################################
class TestInformativeCellsPerVariant:
    # count of cells where fwd > 0 OR rev > 0, per variant.
    @pytest.mark.parametrize("fmt", ["csr", "csc"])
    def test_or_semantics_counts(self, fmt):
        fwd = np.array([[1, 0], [0, 0], [2, 0], [0, 0]])   # 4 cells x 2 vars
        rev = np.array([[0, 0], [3, 0], [1, 0], [0, 0]])
        out = _informative_cells_per_variant(_FORMATS[fmt](fwd), _FORMATS[fmt](rev))
        # var0: c0(f),c1(r),c2(f&r) -> 3 ; var1: none -> 0
        np.testing.assert_array_equal(out, [3.0, 0.0])
    
    def test_both_strands_nonzero_counts_once(self):
        fwd = sp.csc_matrix(np.array([[1], [1]]))
        rev = sp.csc_matrix(np.array([[1], [0]]))
        # c0 has both strands nonzero -> still one informative cell; c1 has fwd only
        np.testing.assert_array_equal(
            _informative_cells_per_variant(fwd, rev), [2.0]
        )
    
    def test_stored_explicit_zero_not_counted(self):
        # fwd has a *stored* zero at cell 0; rev is zero there -> not informative
        fwd = sp.csc_matrix((np.array([0, 4]), (np.array([0, 1]), np.array([0, 0]))), shape=(2, 1))
        rev = sp.csc_matrix((2, 1), dtype=np.float64)
        np.testing.assert_array_equal(
            _informative_cells_per_variant(fwd, rev), [1.0]   # only cell 1
        )
    
    def test_dtype_is_float64(self):
        fwd = sp.csc_matrix(np.eye(3))
        out = _informative_cells_per_variant(fwd, fwd)
        assert out.dtype == np.float64
    
    def test_length_is_nvars(self):
        fwd = sp.csc_matrix(np.ones((6, 4)))
        assert _informative_cells_per_variant(fwd, fwd).shape == (4,)
    
    def test_zero_variants_returns_empty(self):
        empty = sp.csc_matrix((5, 0))
        assert _informative_cells_per_variant(empty, empty).shape == (0,)


###############################################################################
# Testing _strand_concordance
###############################################################################
class TestStrandConcordance:
    @pytest.mark.parametrize("fmt", list(_FORMATS))
    def test_matches_masked_corrcoef_oracle(self, fmt):
        fwd = np.array(
            [[1, 3, 5], [2, 1, 0], [4, 2, 0], [0, 5, 0], [3, 0, 0]], dtype=np.float64
        )
        rev = np.array(
            [[2, 1, 0], [1, 4, 0], [3, 3, 0], [5, 0, 0], [0, 2, 0]], dtype=np.float64
        )
        out = _strand_concordance(_FORMATS[fmt](fwd), _FORMATS[fmt](rev))
        np.testing.assert_allclose(
            out, _concordance_oracle(fwd, rev), rtol=1e-4, atol=1e-5, equal_nan=True
        )
    
    def test_perfect_positive_correlation_is_one(self):
        fwd = sp.csc_matrix(np.array([[1.0], [2.0], [3.0]]))
        rev = sp.csc_matrix(np.array([[2.0], [4.0], [6.0]]))   # rev = 2*fwd
        np.testing.assert_allclose(_strand_concordance(fwd, rev), [1.0], atol=1e-5)
    
    def test_perfect_negative_correlation_is_minus_one(self):
        fwd = sp.csc_matrix(np.array([[1.0], [2.0], [3.0]]))
        rev = sp.csc_matrix(np.array([[6.0], [4.0], [2.0]]))   # rev decreases
        np.testing.assert_allclose(_strand_concordance(fwd, rev), [-1.0], atol=1e-5)
    
    def test_invariant_to_noninformative_cells(self):
        # The core bug fix: dividing moments by informative-cell count, not N.
        # Padding with all-zero (non-informative) cells must not move concordance.
        fwd = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0], [5.0, 0.0]])
        rev = np.array([[2.0, 1.0], [1.0, 3.0], [3.0, 2.0], [0.0, 5.0]])
        base = _strand_concordance(sp.csc_matrix(fwd), sp.csc_matrix(rev))
        pad = np.zeros((500, 2))
        padded = _strand_concordance(
            sp.csc_matrix(np.vstack([fwd, pad])),
            sp.csc_matrix(np.vstack([rev, pad])),
        )
        np.testing.assert_allclose(base, padded, rtol=0, atol=0, equal_nan=True)
    
    def test_fewer_than_two_informative_cells_is_nan(self):
        # one informative cell (fwd>0 at c0), rest zero on both strands
        fwd = sp.csc_matrix(np.array([[5.0], [0.0], [0.0]]))
        rev = sp.csc_matrix(np.array([[0.0], [0.0], [0.0]]))
        assert np.isnan(_strand_concordance(fwd, rev)[0])
    
    def test_zero_variance_strand_is_nan(self):
        # fwd constant across informative cells -> undefined correlation -> NaN
        fwd = sp.csc_matrix(np.array([[2.0], [2.0], [2.0]]))
        rev = sp.csc_matrix(np.array([[1.0], [2.0], [3.0]]))
        assert np.isnan(_strand_concordance(fwd, rev)[0])
    
    def test_shape_mismatch_raises(self):
        fwd = sp.csc_matrix(np.zeros((3, 2)))
        rev = sp.csc_matrix(np.zeros((3, 3)))
        with pytest.raises(ValueError):
            _strand_concordance(fwd, rev)
    
    def test_zero_variants_returns_empty_float32(self):
        out = _strand_concordance(sp.csc_matrix((3, 0)), sp.csc_matrix((3, 0)))
        assert out.shape == (0,)
        assert out.dtype == np.float32
    
    def test_zero_cells_all_nan(self):
        out = _strand_concordance(sp.csc_matrix((0, 2)), sp.csc_matrix((0, 2)))
        assert out.shape == (2,)
        assert np.isnan(out).all()
    
    def test_dtype_is_float32(self):
        fwd = sp.csc_matrix(np.array([[1.0], [2.0], [3.0]]))
        rev = sp.csc_matrix(np.array([[2.0], [1.0], [4.0]]))
        assert _strand_concordance(fwd, rev).dtype == np.float32
    
    def test_sparse_and_dense_agree(self):
        fwd = np.array([[1.0, 0.0], [2.0, 5.0], [3.0, 1.0]])
        rev = np.array([[2.0, 1.0], [1.0, 4.0], [4.0, 2.0]])
        np.testing.assert_allclose(
            _strand_concordance(fwd, rev),
            _strand_concordance(sp.csc_matrix(fwd), sp.csc_matrix(rev)),
            equal_nan=True,
        )


###############################################################################
# Testing _recompute_variant_stats
###############################################################################
class TestRecomputeVariantStats:
    _FWD = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0], [5.0, 0.0]])
    _REV = np.array([[2.0, 1.0], [1.0, 3.0], [3.0, 2.0], [0.0, 5.0]])
    _COV = _FWD + _REV + 1.0   # coverage >= alt everywhere
    
    def test_writes_both_var_columns(self):
        adata = _make_variant_adata(self._FWD, self._REV, self._COV)
        _recompute_variant_stats(adata)
        assert "avg_coverage" in adata.var
        assert "strand_concordance" in adata.var
    
    def test_values_match_standalone_helpers(self):
        adata = _make_variant_adata(self._FWD, self._REV, self._COV)
        _recompute_variant_stats(adata)
        np.testing.assert_allclose(
            adata.var["avg_coverage"].to_numpy(),
            _avg_coverage_per_variant(adata.layers["coverage"]),
        )
        np.testing.assert_allclose(
            adata.var["strand_concordance"].to_numpy(),
            _strand_concordance(adata.layers["alt_fwd"], adata.layers["alt_rev"]),
            equal_nan=True,
        )
    
    def test_returns_same_object(self):
        adata = _make_variant_adata(self._FWD, self._REV, self._COV)
        assert _recompute_variant_stats(adata) is adata
    
    def test_overwrites_stale_var_columns(self):
        adata = _make_variant_adata(self._FWD, self._REV, self._COV)
        adata.var["avg_coverage"] = np.full(adata.n_vars, -999.0)
        adata.var["strand_concordance"] = np.full(adata.n_vars, -999.0)
        _recompute_variant_stats(adata)
        assert (adata.var["avg_coverage"].to_numpy() != -999.0).all()
    
    def test_refreshes_from_current_layers(self):
        # maintain-on-write: after a layer changes, stats must track the new layer.
        adata = _make_variant_adata(self._FWD, self._REV, self._COV)
        _recompute_variant_stats(adata)
        before = adata.var["avg_coverage"].to_numpy().copy()
        
        adata.layers["coverage"] = adata.layers["coverage"] * 2   # mutate the layer
        _recompute_variant_stats(adata)
        after = adata.var["avg_coverage"].to_numpy()
        
        np.testing.assert_allclose(after, 2.0 * before)
        np.testing.assert_allclose(
            after, _avg_coverage_per_variant(adata.layers["coverage"])
        )


###############################################################################
# Testing _safe_div_sparse
###############################################################################
class TestSafeDivSparse:
    # _safe_div_sparse is sparse-only by contract (it calls den.tocsc()), so it is
    # exercised across the sparse formats, not dense.
    @pytest.mark.parametrize("fmt", ["csr", "csc"])
    def test_elementwise_quotient(self, fmt):
        num = np.array([[6.0, 0.0], [0.0, 3.0]])
        den = np.array([[2.0, 0.0], [0.0, 3.0]])
        out = _safe_div_sparse(_FORMATS[fmt](num), _FORMATS[fmt](den))
        np.testing.assert_array_equal(out.toarray(), [[3.0, 0.0], [0.0, 1.0]])
    
    def test_zero_den_where_num_zero_gives_zero_no_inf(self):
        num = sp.csc_matrix(np.array([[6.0, 0.0], [0.0, 3.0]]))
        den = sp.csc_matrix(np.array([[2.0, 0.0], [0.0, 3.0]]))
        arr = _safe_div_sparse(num, den).toarray()
        assert not np.isinf(arr).any() and not np.isnan(arr).any()
    
    def test_num_over_zero_den_raises_with_check(self):
        num = sp.csc_matrix(np.array([[6.0, 0.0], [0.0, 3.0]]))
        den = sp.csc_matrix(np.array([[0.0, 0.0], [0.0, 3.0]]))   # zero under num (0,0)
        with pytest.raises(ValueError):
            _safe_div_sparse(num, den, check=True)
    
    def test_num_over_zero_den_collapses_with_check_false(self):
        num = sp.csc_matrix(np.array([[6.0, 0.0], [0.0, 3.0]]))
        den = sp.csc_matrix(np.array([[0.0, 0.0], [0.0, 3.0]]))
        out = _safe_div_sparse(num, den, check=False).toarray()
        assert out[0, 0] == 0.0                 # silently collapsed, not inf
        assert not np.isinf(out).any() and not np.isnan(out).any()
    
    def test_stored_explicit_zero_den_counts_as_zero(self):
        # den carries a *stored* zero at (0,0); num nonzero there is still div-by-zero
        num = sp.csc_matrix(np.array([[6.0], [3.0]]))
        den = sp.csc_matrix((np.array([0.0, 3.0]), (np.array([0, 1]), np.array([0, 0]))), shape=(2, 1))
        with pytest.raises(ValueError):
            _safe_div_sparse(num, den, check=True)
        # and with check off it must not produce an inf
        out = _safe_div_sparse(num, den, check=False).toarray()
        assert not np.isinf(out).any() and not np.isnan(out).any()
    
    def test_den_nonzero_where_num_zero_is_fine(self):
        # a covered position with no alt reads: legitimate 0, no error either way
        num = sp.csc_matrix(np.array([[0.0], [3.0]]))
        den = sp.csc_matrix(np.array([[4.0], [3.0]]))
        out = _safe_div_sparse(num, den, check=True).toarray()
        np.testing.assert_array_equal(out, [[0.0], [1.0]])
    
    def test_shape_mismatch_raises(self):
        num = sp.csc_matrix(np.zeros((2, 2)))
        den = sp.csc_matrix(np.zeros((2, 3)))
        with pytest.raises(ValueError):
            _safe_div_sparse(num, den)
    
    def test_output_is_sparse_csc(self):
        num = sp.csr_matrix(np.array([[6.0, 0.0], [0.0, 3.0]]))
        den = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 3.0]]))
        out = _safe_div_sparse(num, den)
        assert sp.issparse(out)
        assert out.format == "csc"
    
    def test_out_dtype_respected(self):
        num = sp.csc_matrix(np.array([[6.0], [3.0]]))
        den = sp.csc_matrix(np.array([[2.0], [3.0]]))
        assert _safe_div_sparse(num, den, out_dtype=np.float64).dtype == np.float64


###############################################################################
# Testing read_visium_parquet_map
###############################################################################
def _write_parquet(tmp_path, data, name="map.parquet"):
    path = tmp_path / name
    pl.DataFrame(data).write_parquet(str(path))
    return path


class TestReadVisiumParquetMap:
    def _default_rows(self):
        return {
            "square_002um": ["AAA", "BBB", "CCC", "DDD", "AAA", None],
            "cell_id": [1, 2, 3, 2, 1, 4],          # ints; a duplicate (AAA, 1)
            "in_cell": [True, True, False, True, True, True],
        }
    
    def test_returns_cell_and_cell_id_columns(self, tmp_path):
        p = _write_parquet(tmp_path, self._default_rows())
        out = read_visium_parquet_map(p)
        assert set(out.columns) == {"cell", "cell_id"}
    
    def test_filters_in_cell_false(self, tmp_path):
        p = _write_parquet(tmp_path, self._default_rows())
        out = read_visium_parquet_map(p)
        assert "CCC" not in out["cell"].to_list()   # in_cell == False dropped
    
    def test_filters_null_rows(self, tmp_path):
        p = _write_parquet(tmp_path, self._default_rows())
        out = read_visium_parquet_map(p)
        assert out["cell"].null_count() == 0        # the null square_002um row is gone
    
    def test_cell_id_cast_to_utf8(self, tmp_path):
        p = _write_parquet(tmp_path, self._default_rows())
        out = read_visium_parquet_map(p)
        assert out["cell_id"].dtype == pl.Utf8
    
    def test_deduplicates_pairs(self, tmp_path):
        p = _write_parquet(tmp_path, self._default_rows())
        out = read_visium_parquet_map(p)
        pairs = sorted(zip(out["cell"].to_list(), out["cell_id"].to_list(), strict=True))
        # (AAA,1) appears twice in the source -> one row; (BBB,2) and (DDD,2) both kept
        assert pairs == [("AAA", "1"), ("BBB", "2"), ("DDD", "2")]
    
    def test_cells_include_filters(self, tmp_path):
        p = _write_parquet(tmp_path, self._default_rows())
        out = read_visium_parquet_map(p, cells_include={"2"})
        assert set(out["cell_id"].to_list()) == {"2"}
    
    def test_cells_exclude_filters(self, tmp_path):
        p = _write_parquet(tmp_path, self._default_rows())
        out = read_visium_parquet_map(p, cells_exclude={"2"})
        assert "2" not in out["cell_id"].to_list()
    
    def test_custom_column_names(self, tmp_path):
        rows = {
            "bc": ["AAA", "BBB"],
            "final": [10, 20],
            "keep": [True, False],
        }
        p = _write_parquet(tmp_path, rows, name="custom.parquet")
        out = read_visium_parquet_map(
            p, bam_col="bc", cell_col="final", in_cell_col="keep"
        )
        pairs = sorted(zip(out["cell"].to_list(), out["cell_id"].to_list(), strict=True))
        assert pairs == [("AAA", "10")]            # BBB dropped (keep == False)


###############################################################################
# _utils re-export smoke test
###############################################################################
class TestUtilsReExports:
    # Every public helper added to _utils/basic.py must be re-exported from the
    # package __init__; this catches a missed re-export immediately.
    def test_public_helpers_are_reexported(self):
        import spatial_sigurd_py._utils as u
        import spatial_sigurd_py._utils.basic as b
        
        for name in (
            "_avg_coverage_per_variant",
            "_strand_concordance",
            "_recompute_variant_stats",
            "_to_array",
            "_safe_div_sparse",
            "read_visium_parquet_map",
        ):
            assert hasattr(u, name), f"{name} not re-exported from _utils"
            assert getattr(u, name) is getattr(b, name)
