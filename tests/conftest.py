import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def variant_adata():
    """A small AnnData shaped like the package's real output: cells x variants,
    with the strand/coverage layers the pipeline reads. Function-scoped, so each
    test gets a fresh, independently-mutable object."""
    fwd = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0], [5.0, 0.0]])
    rev = np.array([[2.0, 1.0], [1.0, 3.0], [3.0, 2.0], [0.0, 5.0]])
    cov = fwd + rev + 1.0
    return ad.AnnData(
        X=sp.csr_matrix(fwd.astype(np.float32)),
        layers={
            "alt_fwd": sp.csr_matrix(fwd),
            "alt_rev": sp.csr_matrix(rev),
            "coverage": sp.csr_matrix(cov),
        },
    )
