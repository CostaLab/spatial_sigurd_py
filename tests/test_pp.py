# Tests for the functions in the pp module.

import numpy as np
import pytest
import scipy.sparse as sp

from spatial_sigurd_py.pp import basic as pp_basic
from spatial_sigurd_py.pp.basic import (
    _build_adjacency_1step,
    _estimate_grid_step,
    build_within2_neighborlists,
)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def square_grid(nx, ny, s=1.0):
    """Row-major square lattice of spacing ``s``; index = row * nx + col."""
    xs, ys = np.meshgrid(np.arange(nx) * s, np.arange(ny) * s)
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def hex_grid(nrows, ncols, s=1.0):
    """Offset (Visium-style) hex lattice; every interior spot has 6 neighbors at ``s``."""
    dy = s * np.sqrt(3) / 2
    pts = [
        [c * s + ((s / 2) if r % 2 else 0.0), r * dy]
        for r in range(nrows)
        for c in range(ncols)
    ]
    return np.asarray(pts, dtype=float)


def collinear_chain(n, s=1.0):
    """``n`` evenly spaced points along the x-axis."""
    return np.column_stack([np.arange(n) * s, np.zeros(n)]).astype(float)


def degrees(A):
    """Per-node neighbor count from a binary adjacency matrix."""
    return np.asarray(A.sum(axis=1)).ravel()


def symmetric_adjacency(edges, n):
    """Symmetric, zero-diagonal CSR adjacency from an edge list."""
    A = np.zeros((n, n), dtype=np.int8)
    for a, b in edges:
        A[a, b] = A[b, a] = 1
    np.fill_diagonal(A, 0)
    return sp.csr_matrix(A)


def reference_within2(A):
    """Independent oracle: neighbours within graph distance <= 2, no self."""
    A = sp.csr_matrix(A)
    n = A.shape[0]
    nbr = [set(A.indices[A.indptr[i] : A.indptr[i + 1]].tolist()) for i in range(n)]
    out = []
    for i in range(n):
        reach = set(nbr[i])
        for j in list(nbr[i]):
            reach |= nbr[j]
        reach.discard(i)
        out.append(np.array(sorted(reach), dtype=np.int32))
    return out


def assert_valid_adjacency(A, n):
    """Structural invariants for any non-degenerate adjacency."""
    assert isinstance(A, sp.csr_matrix)
    assert A.shape == (n, n)
    assert A.dtype == np.int8
    assert (A != A.T).nnz == 0          # undirected
    assert A.diagonal().sum() == 0      # no self-loops
    assert set(np.unique(A.data)).issubset({1})  # binary, no stored zeros

def assert_valid_within2(result, n):
    """Structural invariants for any within-2 output (sorted, unique, no self, in-range)."""
    assert isinstance(result, list) and len(result) == n
    for i, arr in enumerate(result):
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.int32
        assert np.all(np.diff(arr) > 0) if arr.size > 1 else True  # sorted + unique
        assert i not in arr.tolist()                               # self excluded
        if arr.size:
            assert arr.min() >= 0 and arr.max() < n

def assert_lists_equal(got, exp):
    assert len(got) == len(exp)
    for i, (g, e) in enumerate(zip(got, exp, strict=True)):
        assert np.array_equal(g, e), f"node {i}: got {g.tolist()} expected {e.tolist()}"

# grid where the two axes have different spacing; median NN distance = min(sx, sy)
_ANISO = np.array(
    [[0.0, 0.0], [0.0, 50.0], [0.0, 100.0],
     [30.0, 0.0], [30.0, 50.0], [30.0, 100.0],
     [60.0, 0.0], [60.0, 50.0], [60.0, 100.0]]
)
_ANISO_OUTLIER = _ANISO.copy()
_ANISO_OUTLIER[2] = [0.0, 100_000.0]  # one far point must not move the median


# --------------------------------------------------------------------------- #
class TestEstimateGridStep:
    @pytest.mark.parametrize(
        "xy, expected",
        [
            (square_grid(3, 3, s=50.0), 50.0),
            (square_grid(4, 4, s=2.5), 2.5),
            (hex_grid(5, 5, s=1.0), 1.0),
            (_ANISO, 30.0),           # nearest neighbor is the tighter (x) axis
            (_ANISO_OUTLIER, 30.0),   # median is robust to a single outlier
        ],
        ids=["square50", "square2.5", "hex", "anisotropic", "anisotropic+outlier"],
    )
    def test_recovers_spacing(self, xy, expected):
        out = _estimate_grid_step(xy)
        assert isinstance(out, float)
        assert out == pytest.approx(expected)
    
    def test_singleton_is_inf(self):
        # k=2 query on a lone point returns inf for the missing neighbor.
        assert _estimate_grid_step(np.array([[0.0, 0.0]])) == np.inf


# --------------------------------------------------------------------------- #
class TestBuildAdjacency1Step:
    @pytest.mark.parametrize(
        "xy",
        [square_grid(4, 4), hex_grid(5, 5), collinear_chain(6),
         np.random.default_rng(0).random((50, 2)) * 10.0],
        ids=["square", "hex", "chain", "random"],
    )
    def test_invariants(self, xy):
        assert_valid_adjacency(_build_adjacency_1step(xy), xy.shape[0])
    
    @pytest.mark.parametrize(
        "idx, expected", [(4, 4), (1, 3), (0, 2)],
        ids=["interior", "edge", "corner"],
    )
    def test_square_grid_orthogonal_degrees(self, idx, expected):
        # rf=1.25 -> r=1.25s: orthogonal (s) in, diagonal (~1.41s) out.
        A = _build_adjacency_1step(square_grid(3, 3), radius_factor=1.25)
        assert degrees(A)[idx] == expected
    
    def test_square_grid_diagonals_at_larger_radius(self):
        # rf=1.5 -> diagonals (~1.41s) included -> 8-connected interior.
        assert degrees(_build_adjacency_1step(square_grid(3, 3), radius_factor=1.5))[4] == 8
    
    def test_collinear_chain_interior_degree_two(self):
        d = degrees(_build_adjacency_1step(collinear_chain(6), radius_factor=1.25))
        assert d[0] == 1 and d[-1] == 1 and np.all(d[1:-1] == 2)
    
    def test_hex_grid_interior_degree_six(self):
        xy = hex_grid(5, 5)
        d = degrees(_build_adjacency_1step(xy, radius_factor=1.25))
        assert d.max() == 6          # second ring (~1.73s) correctly excluded
        assert d[12] == 6            # fully interior spot (row 2, col 2)
    
    def test_radius_factor_too_small_gives_empty_graph(self):
        assert _build_adjacency_1step(square_grid(4, 4), radius_factor=0.5).nnz == 0
    
    def test_nnz_monotone_in_radius_factor(self):
        xy = square_grid(6, 6)
        counts = [_build_adjacency_1step(xy, radius_factor=rf).nnz
                  for rf in (0.9, 1.25, 1.5, 2.1)]
        assert counts == sorted(counts)
    
    def test_duplicate_spot_edge_preserved_no_self_loop(self):
        # 3x3 grid plus a copy of the center spot (index 4) appended as index 9.
        grid = square_grid(3, 3)
        xy = np.vstack([grid, grid[4]])
        A = _build_adjacency_1step(xy, radius_factor=1.25)
        assert_valid_adjacency(A, xy.shape[0])
        assert A[4, 9] == 1 and A[9, 4] == 1     # twins linked by a real edge
        assert A[4, 4] == 0 and A[9, 9] == 0     # not a self-loop
        d = degrees(A)
        assert d[4] == 5 and d[9] == 5           # center's 4 neighbors + the twin
    
    def test_collapsed_median_isolates_distant_spot(self):
        # DOCUMENTS current behavior: mostly-coincident points -> median NN = 0 ->
        # r = 0, so only exactly-coincident spots connect and a far spot is isolated.
        xy = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]])
        assert _estimate_grid_step(xy) == 0.0
        A = _build_adjacency_1step(xy, radius_factor=1.25)
        assert A[0, 1] == 1
        assert degrees(A).tolist() == [1, 1, 0]
    
    def test_single_point_no_crash(self):
        A = _build_adjacency_1step(np.array([[0.0, 0.0]]))
        assert A.shape == (1, 1) and A.nnz == 0
    
    def test_two_points_single_undirected_edge(self):
        A = _build_adjacency_1step(np.array([[0.0, 0.0], [1.0, 0.0]]))
        assert_valid_adjacency(A, 2)
        assert A[0, 1] == 1 and A[1, 0] == 1 and A.nnz == 2
    
    def test_empty_input_warns(self):
        # np.median over an empty slice -> nan (with RuntimeWarning); n == 0 unsupported.
        with pytest.warns(RuntimeWarning):
            _build_adjacency_1step(np.empty((0, 2)))
    
    def test_decoupled_from_estimator(self, monkeypatch):
        # Pin the step so the adjacency logic is tested independently of the estimator.
        monkeypatch.setattr(pp_basic, "_estimate_grid_step", lambda xy: 1.0)
        assert degrees(_build_adjacency_1step(square_grid(3, 3), radius_factor=1.25))[4] == 4


# --------------------------------------------------------------------------- #
class TestBuildWithin2Neighborlists:
    @staticmethod
    def _patch(monkeypatch, A):
        monkeypatch.setattr(pp_basic, "_build_adjacency_1step",
                            lambda xy, radius_factor=1.25: A)
    
    # -- graph logic: mocked adjacency, checked against the reference oracle -- #
    @pytest.mark.parametrize(
        "edges, n",
        [
            ([(0, 1), (1, 2), (2, 3), (3, 4)], 5),                 # path
            ([(0, 1), (1, 2), (0, 2)], 3),                         # triangle
            ([(0, 1), (0, 2), (0, 3), (0, 4)], 5),                 # star
            ([(0, 1)], 3),                                         # isolated node
            ([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], 6), # 2 components
            ([(0, 1), (0, 2), (0, 3)], 5),                         # hub + isolated
        ],
        ids=["path", "triangle", "star", "isolated", "two_components", "hub_isolated"],
    )
    def test_matches_reference(self, monkeypatch, edges, n):
        A = symmetric_adjacency(edges, n)
        self._patch(monkeypatch, A)
        res = build_within2_neighborlists(np.zeros((n, 2)))
        assert_valid_within2(res, n)                         # sortedness, dtype, self-exclusion
        assert_lists_equal(res, reference_within2(A))
    
    @pytest.mark.parametrize("seed", range(10))
    def test_random_graphs_match_reference(self, monkeypatch, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 20))
        M = (rng.random((n, n)) < 0.3).astype(np.int8)
        M = ((M + M.T) > 0).astype(np.int8)
        np.fill_diagonal(M, 0)
        A = sp.csr_matrix(M)
        self._patch(monkeypatch, A)
        res = build_within2_neighborlists(np.zeros((n, 2)))
        assert_valid_within2(res, n)
        assert_lists_equal(res, reference_within2(A))
    
    # -- geometry: real adjacency ------------------------------------------- #
    def test_line(self):
        xy = collinear_chain(6)
        res = build_within2_neighborlists(xy)
        for i in range(6):
            assert res[i].tolist() == sorted(
                k for k in (i - 2, i - 1, i + 1, i + 2) if 0 <= k < 6
            )
    
    def test_square_grid_is_manhattan_ball(self):
        # rf=1.25 -> 4-connectivity -> within-2 == Manhattan distance in {1, 2}.
        N = 5
        xy = square_grid(N, N)
        res = build_within2_neighborlists(xy)
        for cy in range(N):
            for cx in range(N):
                expected = sorted(
                    y * N + x for y in range(N) for x in range(N)
                    if abs(x - cx) + abs(y - cy) in (1, 2)
                )
                assert res[cy * N + cx].tolist() == expected
    
    def test_larger_radius_factor_is_superset(self):
        xy = square_grid(5, 5)
        tight = build_within2_neighborlists(xy, radius_factor=1.25)
        loose = build_within2_neighborlists(xy, radius_factor=1.5)
        for a, b in zip(tight, loose, strict=True):
            assert set(a.tolist()).issubset(set(b.tolist()))
        assert any(b.size > a.size for a, b in zip(tight, loose, strict=True))
    
    def test_separated_clusters_do_not_connect(self):
        block = square_grid(2, 2)
        xy = np.vstack([block, block + np.array([100.0, 0.0])])
        res = build_within2_neighborlists(xy)
        for i in range(4):
            assert set(res[i].tolist()).issubset(set(range(4)))
        for i in range(4, 8):
            assert set(res[i].tolist()).issubset(set(range(4, 8)))
    
    @pytest.mark.parametrize("seed", range(8))
    def test_random_point_clouds_match_reference(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(5, 40))
        xy = rng.random((n, 2)) * 10.0
        res = build_within2_neighborlists(xy)
        A = _build_adjacency_1step(xy, radius_factor=1.25)
        assert_valid_within2(res, n)
        assert_lists_equal(res, reference_within2(A))
    
    # -- edge cases --------------------------------------------------------- #
    @pytest.mark.parametrize(
        "xy, expected",
        [
            (np.array([[0.0, 0.0]]), [[]]),
            (np.array([[0.0, 0.0], [1.0, 0.0]]), [[1], [0]]),
        ],
        ids=["single_point", "two_points"],
    )
    def test_small_inputs(self, xy, expected):
        res = build_within2_neighborlists(xy)
        assert [a.tolist() for a in res] == expected
