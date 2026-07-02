# Tests for the functions in the pp module.

import numpy as np

from spatial_sigurd_py.pp.basic import _estimate_grid_step


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
