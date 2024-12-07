from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class KMeans:

    k: int
    seed: int = 123456789
    tolerance: float = 1.0e-14

    def fit(self, data: NDArray) -> NDArray:

        generator = np.random.default_rng(seed=self.seed)

        # choose k random points from X as centroids
        # assume shape of X is (N, d) where N is number of samples and d is dimension of each sample

        centroids = generator.choice(data, size=self.k, replace=False)
        error = np.inf

        categories = None

        while error >= self.tolerance:

            old_centroids = centroids.copy()
            distances_from_centroids = np.linalg.norm(
                centroids[:, np.newaxis, :] - data[np.newaxis, :, :], axis=2
            )
            categories = np.argmin(distances_from_centroids, axis=0)

            for i in range(self.k):
                centroids[i] = np.mean(data[categories == i])

            error = np.linalg.norm(old_centroids - centroids)

        if categories is not None:
            return categories
        raise ValueError


def spectral_clustering(adjacency_matrix: NDArray, num_categories: int) -> tuple[NDArray, NDArray]:

    rw_laplacian = np.eye(adjacency_matrix.shape[0]) - adjacency_matrix
    eigenvalues, eigenvectors = np.linalg.eig(rw_laplacian)

    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)]
    M = eigenvectors[:, :num_categories]
    U, S, Vh = np.linalg.svd(M.T @ M)

    return M @ U, KMeans(k=num_categories).fit(M)
