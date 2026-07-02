# Tests for the functions in the pp module.

import numpy as np
import pytest
import scipy.sparse as sp

from spatial_sigurd_py.pp.basic import _build_adjacency_1step, _estimate_grid_step


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

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Invariants (geometry-agnostic)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Square grid: known degrees
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Hexagonal (Visium) grid: interior spot has exactly 6 neighbors
# ---------------------------------------------------------------------------
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
 
 
# ---------------------------------------------------------------------------
# radius_factor behavior
# ---------------------------------------------------------------------------
def test_radius_factor_too_small_gives_empty_graph():
    A = _build_adjacency_1step(square_grid(4, 4), radius_factor=0.5)
    assert A.nnz == 0


def test_nnz_monotone_in_radius_factor():
    xy = square_grid(6, 6)
    counts = [_build_adjacency_1step(xy, radius_factor=rf).nnz for rf in (0.9, 1.25, 1.5, 2.1)]
    assert counts == sorted(counts)


# ---------------------------------------------------------------------------
# Duplicate / coincident coordinates
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Small / degenerate n
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Estimator helper, in isolation
# ---------------------------------------------------------------------------
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
