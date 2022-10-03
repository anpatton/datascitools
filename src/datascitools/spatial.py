from sklearn.neighbors import KernelDensity
import numpy as np


class GKR2D:
    """Implementation of Gaussian Kernel Regression.

    Returns:
        GKR object with associated data that be used to calculate and predict.
    """

    def __init__(self, coords: np.array, vals: np.array, b: int):
        self.coords = np.array(coords)
        self.vals = np.array(vals)
        self.b = b

    def __gaussian_kernel(self, z):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2)

    @staticmethod
    def __fast_linalg_norm(a):
        return np.sqrt(np.einsum("ij,ij->i", a, a))

    def __single_predict(self, predict_coord_single):
        kernels = self.__gaussian_kernel(self.__fast_linalg_norm(self.coords - predict_coord_single) / self.b)
        kernel_total = np.sum(kernels)
        x_len = len(self.coords)
        weights = x_len * (kernels / kernel_total)
        result = np.dot(weights.T, self.vals) / len(self.coords)
        return result

    def predict(self, predict_coords: np.array):
        return [self.__single_predict(coord) for coord in predict_coords]


class KDE2D:
    def __init__(self, x: np.array, y: np.array, bw: int, x_grid: np.array, y_grid: np.array):
        self.x = x
        self.y = y
        self.bw = bw
        self.x_grid = x_grid
        self.y_grid = y_grid
        xx, yy = np.meshgrid(x_grid, y_grid)
        self.xy_grid_locations = np.vstack([xx.ravel(), yy.ravel()]).T
        self.xy_event_locations = np.vstack([x, y]).T

    def fit(self):
        kde = KernelDensity(bandwidth=self.bw)
        self.kde = kde.fit(self.xy_event_locations)

    def predict_grid(self):
        dens = np.exp(self.kde.score_samples(self.xy_grid_locations))
        df = pd.DataFrame({"dens": dens})
        df[["x", "y"]] = self.xy_grid_locations
        return df
