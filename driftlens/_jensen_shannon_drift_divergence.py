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
    return E.mean(axis=0)


def matrix_sqrt(X) -> np.ndarray:
    """Computes the square root of a matrix."""
    sqrt_X = scipy.linalg.sqrtm(X)
    return np.real_if_close(sqrt_X)  # Ensure real output for numerical stability


def kl_divergence(mu_x, mu_y, sigma_x, sigma_y) -> float:
    """ Computes the Kullback-Leibler (KL) divergence between two multivariate Gaussian distributions.

    Args:
        mu_x (:obj:`numpy.ndarray`): Mean of the first Gaussian, of shape *(n_features)*.
        mu_y (:obj:`numpy.ndarray`): Mean of the second Gaussian, of shape *(n_features)*.
        sigma_x (:obj:`numpy.ndarray`): Covariance matrix of the first Gaussian, of shape *(n_features, n_features)*.
        sigma_y (:obj:`numpy.ndarray`): Covariance matrix of the second Gaussian, of shape *(n_features, n_features)*.

    Returns:
        :obj:`float`: Kullback-Leibler divergence between the two Gaussian distributions.
    """
    d = mu_x.shape[0]  # Dimensionality

    # Regularization to improve numerical stability
    reg = 1e-6 * np.eye(d)
    sigma_x_reg = sigma_x + reg
    sigma_y_reg = sigma_y + reg

    # Compute log determinant ratio using stable method
    sign_x, logdet_x = np.linalg.slogdet(sigma_x_reg)
    sign_y, logdet_y = np.linalg.slogdet(sigma_y_reg)
    log_det_ratio = logdet_y - logdet_x

    # Compute trace term using stable inversion
    sigma_y_inv = np.linalg.solve(sigma_y_reg, np.eye(d))  # More stable than np.linalg.inv
    trace_term = np.trace(sigma_y_inv @ sigma_x_reg)

    # Compute quadratic term
    mean_diff = mu_y - mu_x
    quadratic_term = mean_diff.T @ sigma_y_inv @ mean_diff

    # Final KL divergence formula
    kl_div = 0.5 * (trace_term + quadratic_term - d + log_det_ratio)
    return kl_div


def jensen_shannon_divergence(mu_x, mu_y, sigma_x, sigma_y) -> float:
    """Computes the Jensen-Shannon divergence between two multivariate Gaussian distributions.

    Args:
        mu_x (:obj:`numpy.ndarray`): Mean of the first Gaussian, of shape *(n_features)*.
        mu_y (:obj:`numpy.ndarray`): Mean of the second Gaussian, of shape *(n_features)*.
        sigma_x (:obj:`numpy.ndarray`): Covariance matrix of the first Gaussian, of shape *(n_features, n_features)*.
        sigma_y (:obj:`numpy.ndarray`): Covariance matrix of the second Gaussian, of shape *(n_features, n_features)*.

    Returns:
        :obj:`float`: Jensen-Shannon divergence between the two Gaussian distributions.
    """
    d = mu_x.shape[0]  # Dimensionality

    # Regularization for stability
    reg = 1e-6 * np.eye(d)
    sigma_x_reg = sigma_x + reg
    sigma_y_reg = sigma_y + reg

    # Compute mixture distribution
    sigma_m = 0.5 * (sigma_x_reg + sigma_y_reg) + reg  # Regularized
    mu_m = 0.5 * (mu_x + mu_y)

    # Compute Jensen-Shannon divergence
    return 0.5 * (kl_divergence(mu_x, mu_m, sigma_x_reg, sigma_m) +
                  kl_divergence(mu_y, mu_m, sigma_y_reg, sigma_m))
