import numpy as np
import scipy.linalg


def get_covariance(E):
    """ Computes the covariance matrix.
    Args:
        E (np.ndarray): Embedding matrix of shape (n_samples, n_features).
    Returns:
        np.ndarray: Covariance matrix of shape (n_features, n_features).
    """
    return np.cov(E, rowvar=False)


def get_mean(E):
    """ Compute the Mean vector.
        Args:
        E (np.ndarray): Embedding matrix of shape (n_samples, n_features).
    Returns:
        np.ndarray: Mean vector of shape (n_features).
    """
    return E.mean(0)


def matrix_sqrt(X):
    """ Computes the square root of a matrix.
        It is a matrix such that sqrt_m @ sqrt_m = X.
    Args:
        X (np.ndarray): Matrix of shape (n_features, n_features).
    Returns:
        np.ndarray: Square root of the matrix.
    """
    return scipy.linalg.sqrtm(X)


def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    """
    Computes the Fréchet distance between multivariate Gaussian distributions x and y, parameterized by their means and covariance matrices.
    Args:
        mu_x (np.ndarray): Mean of the first Gaussian, of shape (n_features).
        mu_y (np.ndarray): Mean of the second Gaussian, of shape (n_features).
        sigma_x (np.ndarray): Covariance matrix of the first Gaussian, of shape (n_features, n_features).
        sigma_y (np.ndarray): Covariance matrix of the second Gaussian, of shape (n_features, n_features).
    Returns:
        float: Fréchet distance between the two Gaussian distributions.
    """
    return np.linalg.norm(mu_x - mu_y) + np.trace(sigma_x + sigma_y - 2*matrix_sqrt(sigma_x @ sigma_y))