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


def mahalanobis_distance(mu_x, mu_y, sigma_x) -> float:
    """ Computes the Mahalanobis distance between two means, assuming covariance sigma_x.

    Args:
        mu_x (:obj:`numpy.ndarray`): Mean of the first Gaussian, of shape *(n_features)*.
        mu_y (:obj:`numpy.ndarray`): Mean of the second Gaussian, of shape *(n_features)*.
        sigma_x (:obj:`numpy.ndarray`): Covariance matrix of the first Gaussian, of shape *(n_features, n_features)*.

    Returns:
        :obj:`float`: Mahalanobis distance between the two means with respect to the covariance of the first distribution.
    """
    # Invert the covariance matrix
    sigma_x_inv = np.linalg.inv(sigma_x)

    # Compute Mahalanobis distance
    diff = mu_x - mu_y
    return np.sqrt(np.dot(np.dot(diff.T, sigma_x_inv), diff))


