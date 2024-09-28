import pytest

import numpy as np

from amosutils.trajectory import RayMinimizer


@pytest.fixture
def cube():
    return np.array(
        [
            [[0, 0, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0]],
            [[0, 0, 1], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0]],
            [[0, 1, 0], [0, 0, 1]],
            [[0, 1, 0], [1, 0, 0]],
            [[1, 0, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0]],
            [[1, 1, 1], [0, 0, 1]],
            [[1, 1, 1], [0, 1, 0]],
            [[1, 1, 1], [1, 0, 0]],
        ]
    )


@pytest.fixture
def square():
    return np.array(
        [
            [[0, 0, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0]],
            [[0, 1, 1], [0, 0, 1]],
            [[0, 1, 1], [0, 1, 0]],
        ]
    )

@pytest.fixture
def cube_minimizer(cube):
    return RayMinimizer(cube[:, 0, :], cube[:, 1, :])


@pytest.fixture
def square_minimizer(square):
    return RayMinimizer(square[:, 0, :], square[:, 1, :])


class TestSquare:
    def test_vertex_lin(self, square, square_minimizer):
        assert square_minimizer.sum_distance(np.array([0, 0, 0])) == pytest.approx(2, rel=1e-12)

    def test_vertex_quad(self, square, square_minimizer):
        assert square_minimizer.sum_quad_distance(np.array([0, 0, 0])) == pytest.approx(2, rel=1e-12)

    def test_centre_lin(self, square, square_minimizer):
        assert square_minimizer.sum_distance(np.array([0, 0.5, 0.5])) == pytest.approx(2, rel=1e-12)

    def test_centre_quad(self, square, square_minimizer):
        assert square_minimizer.sum_quad_distance(np.array([0, 0.5, 0.5])) == pytest.approx(1, rel=1e-12)

    def test_minimize(self, square):
        assert np.allclose(RayMinimizer(square[:, 0, :], square[:, 1, :]).nearest(), np.array([0.0, 0.5, 0.5]))


class TestCube:
    def test_vertex_lin(self, cube, cube_minimizer):
        assert cube_minimizer.sum_distance(np.array([0, 0, 0])) == pytest.approx(6 + 3 * np.sqrt(2), rel=1e-12)

    def test_vertex_quad(self, cube, cube_minimizer):
        assert cube_minimizer.sum_quad_distance(np.array([0, 0, 0])) == pytest.approx(12, rel=1e-12)

    def test_centre_lin(self, cube, cube_minimizer):
        assert cube_minimizer.sum_distance(np.array([0.5, 0.5, 0.5])) == pytest.approx(6 * np.sqrt(2), rel=1e-12)

    def test_centre_quad(self, cube, cube_minimizer):
        assert cube_minimizer.sum_quad_distance(np.array([0.5, 0.5, 0.5])) == pytest.approx(6, rel=1e-12)

    def test_minimize(self, cube):
        assert np.allclose(RayMinimizer(cube[:, 0, :], cube[:, 1, :]).nearest(), np.array([0.5, 0.5, 0.5]))
