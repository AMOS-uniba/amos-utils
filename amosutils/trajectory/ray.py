import numpy as np
import scipy as sp


class RayMinimizer:
    def __init__(self, points: np.ndarray, vectors: np.ndarray):
        assert points.ndim == 2
        assert vectors.ndim == 2
        assert points.shape == vectors.shape
        self.points = points
        self.vectors = vectors

    def sum_distance(self, point: np.array) -> float:
        p = point - self.points
        dist = np.linalg.norm(np.cross(p, self.vectors), axis=1, ord=2) / np.linalg.norm(self.vectors, axis=1, ord=2)
        return np.sum(dist)

    def sum_quad_distance(self, point: np.ndarray) -> float:
        p = point - self.points
        dist = np.linalg.norm(np.cross(p, self.vectors), axis=1, ord=2) / np.linalg.norm(self.vectors, axis=1, ord=2)
        return np.sum(np.square(dist))

    def nearest(self) -> np.ndarray:
        result = sp.optimize.minimize(self.sum_quad_distance, np.array([0, 0, 0]).T, method="L-BFGS-B")
        return result.x