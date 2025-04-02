import numpy as np
import scipy.linalg

def get_covariance(E) -> np.ndarray:
    """ Computes the covariance matrix."""
    return np.cov(E, rowvar=False)

def get_mean(E) -> np.ndarray:
    """ Compute the Mean vector."""
    return E.mean(axis=0)

def matrix_sqrt(X) -> np.ndarray:
    """ Computes the square root of a matrix."""
    sqrt_X = scipy.linalg.sqrtm(X)
    return np.real_if_close(sqrt_X)  # Ensure real output for numerical stability

def kl_divergence(mu_x, mu_y, sigma_x, sigma_y) -> float:
    """ Computes the Kullback-Leibler (KL) divergence between two multivariate Gaussian distributions."""
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

