import numpy as np
from bdmtx.roi.geometry import quad_area, is_convex_quad, expand_quad


def test_quad_area_and_convexity():
    quad = np.array([[0.0,0.0],[2.0,0.0],[2.0,1.0],[0.0,1.0]])
    assert abs(quad_area(quad) - 2.0) < 1e-6
    assert is_convex_quad(quad)


def test_expand_quad():
    quad = np.array([[0.0,0.0],[2.0,0.0],[2.0,1.0],[0.0,1.0]])
    e = expand_quad(quad, factor=1.5)
    center = quad.mean(axis=0)
    assert e.shape == quad.shape
    assert (e - center).ptp() > (quad - center).ptp()
