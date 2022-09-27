from sklearn.neighbors import KernelDensity
import numpy as np

class GKR2D:
    """Implementation of Gaussian Kernel Regression.

    Returns:
        GKR object with associated data that be used to calculate and predict.
    """

    def __init__(self, x: np.array, y: np.array, b: int):
        self.x = np.array(x)
        self.y = np.array(y)
        self.b = b

    '''Implement the Gaussian Kernel'''
    def gaussian_kernel(self, z):
        return (1/np.sqrt(2*np.pi))*np.exp(-0.5*z**2)

    '''Calculate weights and return prediction'''
    def predict(self, X):
        kernels = np.array([self.gaussian_kernel((np.linalg.norm(xi-X))/self.b) for xi in self.x])
        weights = np.array([len(self.x) * (kernel/np.sum(kernels)) for kernel in kernels])
        return np.dot(weights.T, self.y)/len(self.x)

class KDE2D:
    def __init__(self, x: np.array, y: np.array, bw: int, x_grid: np.array,y_grid: np.array):
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
        df = pd.DataFrame({'dens': dens})
        df[['x','y']] = self.xy_grid_locations
        return df