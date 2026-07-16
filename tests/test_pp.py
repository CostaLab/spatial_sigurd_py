# Tests for the functions in the pp module.

import numpy as np
import pytest
import scipy.sparse as sp

from spatial_sigurd_py.pp import basic as pp_basic
from spatial_sigurd_py.pp.basic import _build_adjacency_1step, _estimate_grid_step, build_within2_neighborlists


###############################################################################
# Testing _estimate_grid_step
###############################################################################
class TestEstimateGridStep:
    def test_median50_result(self):
        data_xy = np.array([[  0.0,   0.0], [  0.0,  50.0], [  0.0, 100.0],
                            [ 50.0,   0.0], [ 50.0,  50.0], [ 50.0, 100.0],
                            [100.0,   0.0], [100.0,  50.0], [100.0, 100.0]])
        
        out = _estimate_grid_step(data_xy)
        assert isinstance(out, float)
        assert np.isclose(out, 50)
    
    def test_median30_result(self):
        data_xy = np.array([[ 0.0,   0.0], [ 0.0, 50.0], [ 0.0, 100.0],
                            [30.0,   0.0], [30.0, 50.0], [30.0, 100.0],
                            [60.0,   0.0], [60.0, 50.0], [60.0, 100.0]])
        
        out = _estimate_grid_step(data_xy)
        assert isinstance(out, float)
        assert np.isclose(out, 30)
    
    def test_median30_with_outlier_result(self):
        data_xy = np.array([[ 0.0,   0.0], [ 0.0, 50.0], [ 0.0, 100000.0],
                            [30.0,   0.0], [30.0, 50.0], [30.0,    100.0],
                            [60.0,   0.0], [60.0, 50.0], [60.0,    100.0]])
        
        out = _estimate_grid_step(data_xy)
        assert isinstance(out, float)
        assert np.isclose(out, 30)

###############################################################################
# Testing _build_adjacency_1step
###############################################################################
"""Tests for the spatial 1-step adjacency builder.

Covers ``_build_adjacency_1step`` and its coupled helper ``_estimate_grid_step``
in ``spatial_sigurd_py.pp.basic``.

Because the neighbor radius is derived from the data (``radius_factor`` times the
median nearest-neighbor distance), the correctness tests use regular lattices
where that median is unambiguous, and the edge-case tests target inputs that make
the median degenerate.
"""

# Geometry helpers
def square_grid(nx: int, ny: int, s: float = 1.0) -> np.ndarray:
    """Row-major square lattice of spacing ``s``; index = row * nx + col."""
    xs, ys = np.meshgrid(np.arange(nx) * s, np.arange(ny) * s)
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)

def hex_grid(nrows: int, ncols: int, s: float = 1.0) -> np.ndarray:
    """Offset (Visium-style) hexagonal lattice with nearest-neighbor distance ``s``.
    
    Odd rows are shifted right by ``s / 2``; row pitch is ``s * sqrt(3) / 2`` so
    every spot has 6 neighbors at distance exactly ``s``. Row-major:
    index = row * ncols + col.
    """
    dy = s * np.sqrt(3) / 2
    pts = []
    for r in range(nrows):
        offset = (s / 2) if (r % 2) else 0.0
        for c in range(ncols):
            pts.append([c * s + offset, r * dy])
    return np.asarray(pts, dtype=float)

def collinear_chain(n: int, s: float = 1.0) -> np.ndarray:
    """``n`` evenly spaced points along the x-axis."""
    return np.column_stack([np.arange(n) * s, np.zeros(n)]).astype(float)

def degrees(A: sp.spmatrix) -> np.ndarray:
    """Per-node neighbor count from a binary adjacency matrix."""
    return np.asarray(A.sum(axis=1)).ravel()

def assert_valid_adjacency(A: sp.spmatrix, n: int) -> None:
    """Structural invariants that must hold for any non-degenerate input."""
    assert isinstance(A, sp.csr_matrix)
    assert A.shape == (n, n)
    assert A.dtype == np.int8
    # Undirected: the (A + A.T) symmetrization must survive.
    assert (A != A.T).nnz == 0
    # No self-loops.
    assert A.diagonal().sum() == 0
    # Binary, and eliminate_zeros() left no explicit zeros stored.
    assert set(np.unique(A.data)).issubset({1})
    assert bool((A.data != 0).all())

# Invariants (geometry-agnostic)
@pytest.mark.parametrize(
    "xy",
    [
        square_grid(4, 4),
        hex_grid(5, 5),
        collinear_chain(6),
        np.random.default_rng(0).random((50, 2)) * 10.0,
    ],
    ids=["square", "hex", "chain", "random"],
)
def test_invariants(xy):
    A = _build_adjacency_1step(xy)
    assert_valid_adjacency(A, xy.shape[0])

# Square grid: known degrees
def test_square_grid_degrees_orthogonal_only():
    # rf=1.25 -> r=1.25s: orthogonal (s) in, diagonal (s*sqrt(2)~=1.41s) out.
    A = _build_adjacency_1step(square_grid(3, 3), radius_factor=1.25)
    d = degrees(A)
    assert d[4] == 4  # interior
    assert d[1] == 3  # edge
    assert d[0] == 2  # corner

def test_square_grid_degrees_with_diagonals():
    # rf=1.5 -> r=1.5s: diagonals (1.41s) now included -> 8-connected.
    A = _build_adjacency_1step(square_grid(3, 3), radius_factor=1.5)
    assert degrees(A)[4] == 8

def test_collinear_chain_interior_degree_two():
    A = _build_adjacency_1step(collinear_chain(6), radius_factor=1.25)
    d = degrees(A)
    assert d[0] == 1 and d[-1] == 1
    assert np.all(d[1:-1] == 2)

# Hexagonal (Visium) grid: interior spot has exactly 6 neighbors
def test_hex_grid_interior_degree_six():
    xy = hex_grid(5, 5)
    # median NN distance recovers the true step (all spots have a distance-s neighbor).
    assert _estimate_grid_step(xy) == pytest.approx(1.0)
    
    A = _build_adjacency_1step(xy, radius_factor=1.25)
    d = degrees(A)
    
    # No spot exceeds 6: the second ring at sqrt(3)*s ~= 1.73s is correctly excluded.
    assert d.max() == 6
    assert (d == 6).any()
    # Geometric center of a 5x5 offset grid (row 2, col 2) is fully interior.
    assert d[12] == 6

# radius_factor behavior
def test_radius_factor_too_small_gives_empty_graph():
    A = _build_adjacency_1step(square_grid(4, 4), radius_factor=0.5)
    assert A.nnz == 0

def test_nnz_monotone_in_radius_factor():
    xy = square_grid(6, 6)
    counts = [_build_adjacency_1step(xy, radius_factor=rf).nnz for rf in (0.9, 1.25, 1.5, 2.1)]
    assert counts == sorted(counts)

# Duplicate / coincident coordinates
def test_duplicate_spot_edge_preserved_no_self_loop():
    # 3x3 grid plus a copy of the center spot (index 4) appended as index 9.
    grid = square_grid(3, 3)
    xy = np.vstack([grid, grid[4]])
    
    # A single coincident pair does not move the median step.
    assert _estimate_grid_step(xy) == pytest.approx(1.0)
    
    A = _build_adjacency_1step(xy, radius_factor=1.25)
    assert_valid_adjacency(A, xy.shape[0])
    
    # The twins are distinct indices at distance 0 -> a real edge, not a self-loop.
    assert A[4, 9] == 1 and A[9, 4] == 1
    assert A[4, 4] == 0 and A[9, 9] == 0
    # Each twin sees the center's 4 orthogonal neighbors plus the other twin.
    d = degrees(A)
    assert d[4] == 5 and d[9] == 5

def test_collapsed_median_isolates_distant_spot():
    # DOCUMENTS current behavior: when most points are coincident the median NN
    # distance is 0, so r = 0 and only exactly-coincident spots are connected.
    # A well-separated spot ends up isolated. Flagging in case that ever needs
    # to become a guarded/degenerate-input error instead.
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]])
    assert _estimate_grid_step(xy) == 0.0
    
    A = _build_adjacency_1step(xy, radius_factor=1.25)
    assert_valid_adjacency(A, 3)
    assert A[0, 1] == 1  # coincident pair still linked
    assert degrees(A).tolist() == [1, 1, 0]  # distant spot isolated

# Small / degenerate n
def test_single_point_no_crash():
    A = _build_adjacency_1step(np.array([[0.0, 0.0]]))
    assert A.shape == (1, 1)
    assert A.nnz == 0

def test_two_points_single_undirected_edge():
    A = _build_adjacency_1step(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert_valid_adjacency(A, 2)
    assert A[0, 1] == 1 and A[1, 0] == 1
    assert A.nnz == 2  # both directions stored for one undirected edge

def test_empty_input_unsupported():
    # np.median over an empty slice -> nan (with a RuntimeWarning). Guard against
    # silently building a graph from a nan radius; treat n == 0 as unsupported.
    with pytest.warns(RuntimeWarning):
        _build_adjacency_1step(np.empty((0, 2)))

# Estimator helper, in isolation
def test_estimate_grid_step_recovers_spacing():
    assert _estimate_grid_step(square_grid(4, 4, s=2.5)) == pytest.approx(2.5)

def test_estimate_grid_step_singleton_is_inf():
    # k=2 query on a lone point returns inf for the missing neighbor.
    assert _estimate_grid_step(np.array([[0.0, 0.0]])) == np.inf


def test_adjacency_decoupled_from_estimator(monkeypatch):
    # Pin the step so the adjacency logic is tested independently of the
    # estimator: force r = 1.25 * 1.0 on a unit square grid.
    monkeypatch.setattr(
        "spatial_sigurd_py.pp.basic._estimate_grid_step",
        lambda xy: 1.0,
    )
    A = _build_adjacency_1step(square_grid(3, 3), radius_factor=1.25)
    assert degrees(A)[4] == 4

###############################################################################
# Testing _build_adjacency_1step
###############################################################################
"""Tests for ``build_within2_neighborlists``.

The function returns, for every point ``i``, the set of points reachable within
graph distance <= 2 on the 1-step spatial adjacency graph, excluding ``i`` itself.

Two complementary strategies are used:

* **Graph-logic tests** monkeypatch ``_build_adjacency_1step`` so an arbitrary
  adjacency matrix can be fed in. These pin down the BFS-to-depth-2 traversal
  independently of any geometry, and compare against a plain reference BFS.
* **Geometry tests** exercise the real ``_build_adjacency_1step`` on layouts
  (a line, a square grid, random clouds) where the neighbourhoods are known or
  can be cross-checked against the reference.

Adjust the import below if your package name / layout differs; everything else
patches ``pp_basic._build_adjacency_1step`` by reference.
"""

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def reference_within2(A) -> list[np.ndarray]:
    """Independent reference: neighbours within graph distance <= 2, no self."""
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


def symmetric_adjacency(edges: list[tuple[int, int]], n: int) -> sp.csr_matrix:
    """Build a symmetric, zero-diagonal CSR adjacency from an edge list."""
    A = np.zeros((n, n), dtype=np.int8)
    for a, b in edges:
        A[a, b] = A[b, a] = 1
    np.fill_diagonal(A, 0)
    return sp.csr_matrix(A)


def patch_adjacency(monkeypatch, A: sp.csr_matrix) -> None:
    """Make ``build_within2_neighborlists`` use ``A`` regardless of ``xy``."""
    A = sp.csr_matrix(A)
    monkeypatch.setattr(
        pp_basic,
        "_build_adjacency_1step",
        lambda xy, radius_factor=1.25: A,
    )


def assert_valid_output(result, n: int) -> None:
    """Structural invariants that must hold for any valid output."""
    assert isinstance(result, list)
    assert len(result) == n
    for i, arr in enumerate(result):
        assert isinstance(arr, np.ndarray)
        assert np.issubdtype(arr.dtype, np.integer)
        # sorted strictly ascending => also implies uniqueness
        assert np.all(np.diff(arr) > 0) if arr.size > 1 else True
        # indices in range and self excluded
        if arr.size:
            assert arr.min() >= 0 and arr.max() < n
        assert i not in arr.tolist()


def assert_lists_equal(got, exp) -> None:
    assert len(got) == len(exp)
    for i, (g, e) in enumerate(zip(got, exp, strict=True)):
        assert np.array_equal(g, e), f"node {i}: got {g.tolist()} expected {e.tolist()}"


# --------------------------------------------------------------------------- #
# Graph-logic tests (adjacency mocked, geometry irrelevant)
# --------------------------------------------------------------------------- #
def test_path_graph(monkeypatch):
    # 0-1-2-3-4 ; within-2 is {i-2, i-1, i+1, i+2} clipped to range
    A = symmetric_adjacency([(0, 1), (1, 2), (2, 3), (3, 4)], n=5)
    patch_adjacency(monkeypatch, A)
    res = build_within2_neighborlists(np.zeros((5, 2)))
    expected = [
        np.array([1, 2], dtype=np.int32),
        np.array([0, 2, 3], dtype=np.int32),
        np.array([0, 1, 3, 4], dtype=np.int32),
        np.array([1, 2, 4], dtype=np.int32),
        np.array([2, 3], dtype=np.int32),
    ]
    assert_valid_output(res, 5)
    assert_lists_equal(res, expected)


def test_triangle_excludes_self(monkeypatch):
    # In a triangle, node i is reachable from itself in 2 hops; it must still
    # be excluded from its own within-2 list.
    A = symmetric_adjacency([(0, 1), (1, 2), (0, 2)], n=3)
    patch_adjacency(monkeypatch, A)
    res = build_within2_neighborlists(np.zeros((3, 2)))
    assert_lists_equal(
        res,
        [
            np.array([1, 2], dtype=np.int32),
            np.array([0, 2], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
        ],
    )


def test_star_graph(monkeypatch):
    # Center 0 connected to leaves 1..4. Center reaches leaves in 1 hop;
    # each leaf reaches the center (1 hop) and every other leaf (2 hops).
    A = symmetric_adjacency([(0, 1), (0, 2), (0, 3), (0, 4)], n=5)
    patch_adjacency(monkeypatch, A)
    res = build_within2_neighborlists(np.zeros((5, 2)))
    assert res[0].tolist() == [1, 2, 3, 4]
    for leaf in (1, 2, 3, 4):
        assert res[leaf].tolist() == [x for x in range(5) if x != leaf]


def test_isolated_node(monkeypatch):
    # Node 2 has no edges -> empty list; others unaffected.
    A = symmetric_adjacency([(0, 1)], n=3)
    patch_adjacency(monkeypatch, A)
    res = build_within2_neighborlists(np.zeros((3, 2)))
    assert res[2].tolist() == []
    assert res[0].tolist() == [1]
    assert res[1].tolist() == [0]


def test_disconnected_components_do_not_leak(monkeypatch):
    # Two separate triangles: {0,1,2} and {3,4,5}. No index should cross over.
    A = symmetric_adjacency(
        [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], n=6
    )
    patch_adjacency(monkeypatch, A)
    res = build_within2_neighborlists(np.zeros((6, 2)))
    for i in (0, 1, 2):
        assert set(res[i].tolist()).issubset({0, 1, 2})
    for i in (3, 4, 5):
        assert set(res[i].tolist()).issubset({3, 4, 5})


def test_symmetry_property(monkeypatch):
    # within-2 is symmetric: j in within2[i]  <=>  i in within2[j].
    A = symmetric_adjacency([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], n=4)
    patch_adjacency(monkeypatch, A)
    res = build_within2_neighborlists(np.zeros((4, 2)))
    for i in range(4):
        for j in res[i].tolist():
            assert i in res[j].tolist()


@pytest.mark.parametrize("seed", range(10))
def test_random_graphs_match_reference(monkeypatch, seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 20))
    M = (rng.random((n, n)) < 0.3).astype(np.int8)
    M = ((M + M.T) > 0).astype(np.int8)
    np.fill_diagonal(M, 0)
    A = sp.csr_matrix(M)
    patch_adjacency(monkeypatch, A)
    res = build_within2_neighborlists(np.zeros((n, 2)))
    assert_valid_output(res, n)
    assert_lists_equal(res, reference_within2(A))


def test_no_leftover_state_between_iterations(monkeypatch):
    # The internal scratch buffer is reused; a dense hub followed by an isolated
    # node would expose any failure to reset it.
    A = symmetric_adjacency([(0, 1), (0, 2), (0, 3)], n=5)  # node 4 isolated
    patch_adjacency(monkeypatch, A)
    res = build_within2_neighborlists(np.zeros((5, 2)))
    assert res[4].tolist() == []
    assert_lists_equal(res, reference_within2(A))


# --------------------------------------------------------------------------- #
# Geometry tests (real _build_adjacency_1step)
# --------------------------------------------------------------------------- #
def test_geometry_line():
    # Equally spaced points on a line: within-2 is {i-2, i-1, i+1, i+2}.
    xy = np.array([[float(i), 0.0] for i in range(6)])
    res = build_within2_neighborlists(xy)
    for i in range(6):
        expected = sorted(
            k for k in (i - 2, i - 1, i + 1, i + 2) if 0 <= k < 6
        )
        assert res[i].tolist() == expected


def test_geometry_square_grid_is_manhattan_ball():
    # On a unit square grid with the default factor 1.25, only orthogonal
    # neighbours connect (diagonals are ~1.414 away), so within-2 equals the
    # set of points at Manhattan distance 1 or 2.
    N = 5
    xy = np.array([[float(x), float(y)] for y in range(N) for x in range(N)])
    res = build_within2_neighborlists(xy)

    def manhattan_ball(cx, cy):
        return sorted(
            y * N + x
            for y in range(N)
            for x in range(N)
            if abs(x - cx) + abs(y - cy) in (1, 2)
        )

    for cy in range(N):
        for cx in range(N):
            assert res[cy * N + cx].tolist() == manhattan_ball(cx, cy)


def test_radius_factor_too_small_gives_no_neighbours():
    # factor < 1 => search radius below the grid step => no edges => all empty.
    xy = np.array([[float(i), 0.0] for i in range(5)])
    res = build_within2_neighborlists(xy, radius_factor=0.5)
    assert all(arr.size == 0 for arr in res)


def test_larger_radius_factor_is_superset():
    # Including diagonals (factor 1.5) can only add reachable points on a grid.
    N = 5
    xy = np.array([[float(x), float(y)] for y in range(N) for x in range(N)])
    tight = build_within2_neighborlists(xy, radius_factor=1.25)
    loose = build_within2_neighborlists(xy, radius_factor=1.5)
    for a, b in zip(tight, loose, strict=True):
        assert set(a.tolist()).issubset(set(b.tolist()))
    # and strictly more somewhere (diagonals now connect)
    assert any(b.size > a.size for a, b in zip(tight, loose, strict=True))


def test_two_separated_clusters_do_not_connect():
    # Two tight 2x2 blocks placed far apart; no within-2 crosses clusters.
    block = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    xy = np.vstack([block, block + np.array([100.0, 0.0])])
    res = build_within2_neighborlists(xy)
    left, right = set(range(4)), set(range(4, 8))
    for i in range(4):
        assert set(res[i].tolist()).issubset(left)
    for i in range(4, 8):
        assert set(res[i].tolist()).issubset(right)


@pytest.mark.parametrize("seed", range(8))
def test_random_point_clouds_match_reference(seed):
    # End-to-end: the traversal output equals a reference BFS run on the same
    # adjacency the function itself builds.
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 40))
    xy = rng.random((n, 2)) * 10.0
    res = build_within2_neighborlists(xy)
    A = pp_basic._build_adjacency_1step(xy, radius_factor=1.25)
    assert_valid_output(res, n)
    assert_lists_equal(res, reference_within2(A))


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_single_point():
    res = build_within2_neighborlists(np.array([[0.0, 0.0]]))
    assert len(res) == 1
    assert res[0].tolist() == []


def test_two_points_close():
    res = build_within2_neighborlists(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert res[0].tolist() == [1]
    assert res[1].tolist() == [0]


def test_output_dtype_is_int32(monkeypatch):
    A = symmetric_adjacency([(0, 1), (1, 2)], n=3)
    patch_adjacency(monkeypatch, A)
    res = build_within2_neighborlists(np.zeros((3, 2)))
    for arr in res:
        assert arr.dtype == np.int32
