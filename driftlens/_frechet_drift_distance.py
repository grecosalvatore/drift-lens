import numpy as np
import scipy.linalg


def get_covariance(E) -> np.ndarray:
    """ Computes the covariance matrix.

    Args:
        E (:obj:`numpy.ndarray`): Embedding matrix of shape *(n_samples, n_features)*.

    Returns:
        :obj:`numpy.ndarray` Covariance matrix of shape *(n_features, n_features)*.
    """
    return np.cov(E, rowvar=False)


def get_mean(E) -> np.ndarray:
    """ Compute the Mean vector.

    Args:
        E (:obj:`numpy.ndarray`): Embedding matrix of shape *(n_samples, n_features)*.
        
    Returns:
        :obj:`numpy.ndarray`: Mean vector of shape *(n_features)*.
    """
    return E.mean(0)


def matrix_sqrt(X) -> np.ndarray:
    """ Computes the square root of a matrix. It is a matrix such that sqrt_m @ sqrt_m = X.
    
    Args:
        X (:obj:`numpy.ndarray`): Matrix of shape *(n_features, n_features)*.

    Returns:
        :obj:`numpy.ndarray`: Square root of the matrix.
    """
    return scipy.linalg.sqrtm(X)


def frechet_distance(mu_x, mu_y, sigma_x, sigma_y) -> float:
    """ Computes the Fréchet distance between multivariate Gaussian distributions.

    Args:
        mu_x        (:obj:`numpy.ndarray`): Mean of the first Gaussian, of shape *(n_features)*.
        mu_y        (:obj:`numpy.ndarray`): Mean of the second Gaussian, of shape *(n_features)*.
        sigma_x     (:obj:`numpy.ndarray`): Covariance matrix of the first Gaussian, of shape *(n_features, n_features)*.
        sigma_y     (:obj:`numpy.ndarray`): Covariance matrix of the second Gaussian, of shape *(n_features, n_features)*.

    Returns:
        :obj:`float`: Fréchet distance between the two Gaussian distributions.
    """
    return np.linalg.norm(mu_x - mu_y) + np.trace(sigma_x + sigma_y - 2*matrix_sqrt(sigma_x @ sigma_y))