import numpy as np
import scipy.linalg


def get_covariance(E):
    """ Computes the covariance matrix. """
    return np.cov(E, rowvar=False)


def get_mean(E):
    """ Compute the Mean vector. """
    return E.mean(0)


def matrix_sqrt(X):
    """
    Computes the square root of a matrix. It is a matrix such that sqrt_m @ sqrt_m = X.
    """
    return scipy.linalg.sqrtm(X)


def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    """
    Computes the Fréchet distance between multivariate Gaussians,
    parameterized by their means and covariance matrices.
    Parameters:
        mu_x: the mean of the first Gaussian, (n_features)
        mu_y: the mean of the second Gaussian, (n_features)
        sigma_x: the covariance matrix of the first Gaussian, (n_features, n_features)
        sigma_y: the covariance matrix of the second Gaussian, (n_features, n_features)
    """
    return np.linalg.norm(mu_x - mu_y) + np.trace(sigma_x + sigma_y - 2*matrix_sqrt(sigma_x @ sigma_y))