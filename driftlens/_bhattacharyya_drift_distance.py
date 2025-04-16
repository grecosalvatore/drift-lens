import numpy as np


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
        :obj:`numpy.ndarray`: Mean vector of shape *(n_features,)*.
    """
    return E.mean(axis=0)


def bhattacharyya_distance(mu_x, mu_y, sigma_x, sigma_y) -> float:
    """ Computes the Bhattacharyya distance between two multivariate Gaussian distributions.

    Args:
        mu_x (:obj:`numpy.ndarray`): Mean of the first Gaussian, of shape *(n_features,)*.
        mu_y (:obj:`numpy.ndarray`): Mean of the second Gaussian, of shape *(n_features,)*.
        sigma_x (:obj:`numpy.ndarray`): Covariance matrix of the first Gaussian, of shape *(n_features, n_features)*.
        sigma_y (:obj:`numpy.ndarray`): Covariance matrix of the second Gaussian, of shape *(n_features, n_features)*.

    Returns:
        :obj:`float`: Bhattacharyya distance between the two Gaussian distributions.
    """
    sigma_avg = 0.5 * (sigma_x + sigma_y)
    mean_diff = mu_y - mu_x

    # Term 1: Mahalanobis-like term
    term1 = 0.125 * mean_diff.T @ np.linalg.solve(sigma_avg, mean_diff)

    # Term 2: Log determinant term
    sign_x, logdet_x = np.linalg.slogdet(sigma_x)
    sign_y, logdet_y = np.linalg.slogdet(sigma_y)
    sign_avg, logdet_avg = np.linalg.slogdet(sigma_avg)

    term2 = 0.5 * (logdet_avg - 0.5 * (logdet_x + logdet_y))

    return term1 + term2
