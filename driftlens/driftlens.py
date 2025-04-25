import os.path

from typing import List, Tuple

from driftlens.distribution_distances import frechet_drift_distance as fdd
from driftlens.distribution_distances import mahalanobis_drift_distance as mdd
from driftlens.distribution_distances import kullback_leibler_drift_divergence as kldd
from driftlens.distribution_distances import bhattacharyya_drift_distance as bdd
from driftlens.distribution_distances import jensen_shannon_drift_divergence as jsdd
from driftlens import _baseline as _baseline
from driftlens import _threshold as _threshold

import matplotlib.pyplot as plt
from driftlens import _utils as utils

import numpy as np
import pandas as pd
from scipy import stats


class DriftLens:
    """ DriftLens Class.

    Attributes:
        baseline                (:obj:`BaselineClass`): BaselineClass object.
        threshold               (:obj:`ThresholdClass`): ThresholdClass object.
        label_list              (:obj:`list(str)`): List of class labels.
        batch_n_pc              (:obj:`int`): Number of principal components to use for the per-batch.
        per_label_n_pc          (:obj:`int`): Number of principal components to use for the per-label.
        baseline_algorithms     (:obj:`dict`): Dictionary of possible baseline algorithms.
        threshold_estimators    (:obj:`dict`): Dictionary of possible threshold estimators.
    """
    def __init__(self, label_list=None):

        self.baseline = None  # BaselineClass object
        self.threshold = None  # ThresholdClass object

        self.label_list = label_list  # List of class labels
        self.batch_n_pc = None  # Number of principal components to use for the per-batch
        self.per_label_n_pc = None  # Number of principal components to use for the per-label

        self.baseline_algorithms = {"StandardBaselineEstimator": "Description"}
        self.threshold_estimators = {"KFoldThresholdEstimator": "Description"}

    def estimate_baseline(self,
                          E: np.ndarray,
                          Y: np.ndarray,
                          label_list: List[int],
                          batch_n_pc: int,
                          per_label_n_pc: int,
                          baseline_algorithm: str = "StandardBaselineEstimator"
                          ) -> _baseline.BaselineClass:
        r""" Estimates the baseline.

        Args:
            label_list          (:obj:`list(int)`): List of class label ids used to train the model.
            batch_n_pc          (:obj:`int`): Number of principal components to use for the per-batch.
            per_label_n_pc      (:obj:`int`): Number of principal components to use for the per-label.
            E                   (:obj:`numpy.ndarray`): Embedding matrix of shape *(m, d)*, where *m* is the number of samples and *d* the embedding dimensionality.
            Y                   (:obj:`numpy.ndarray`): Vector of predicted labels of shape *(m, 1)*, where m is the number of samples.
            baseline_algorithm  (:obj:`str`, `optional`): Baseline estimation algorithm to use. Possible values are: *"StandardBaselineEstimator"*. If not provided, the default value is *"StandardBaselineEstimator"*.

        Returns:
            :class:`~driftlens._baseline.BaselineClass`: An instance of the `BaselineClass` class from the `_baseline.py` module, performing the offline phase of DriftLens.
        """

        self.label_list = label_list
        self.batch_n_pc = batch_n_pc
        self.per_label_n_pc = per_label_n_pc

        # Choose the selected baseline estimator algorithm
        if baseline_algorithm in self.baseline_algorithms.keys():
            if baseline_algorithm == "StandardBaselineEstimator":
                baseline_estimator = _baseline.StandardBaselineEstimator(self.label_list, self.batch_n_pc, self.per_label_n_pc)
        else:
            raise Exception("Unknown baseline algorithm. Call the 'baseline_algorithms' attribute to read possible baseline estimation algorithms.")

        # Execute the baseline estimation
        try:
            self.baseline = baseline_estimator.estimate_baseline(E, Y)
        except Exception as e:
            raise Exception(f'Error in creating the baseline: {e}')

        return self.baseline

    def save_baseline(self, folder_path: str, baseline_name: str) -> str:
        """ Stores persistently on disk the baseline.

        Args:
            folder_path     (:obj:`str`): Folder path where save the baseline.
            baseline_name   (:obj:`str`): Filename of the baseline folder.

        Returns:
            :obj:`str`: Baseline folder path.
        """
        if self.baseline is not None:
            baseline_path = self.baseline.save(folder_path, baseline_name)
        else:
            raise Exception(f'Error: Baseline has not yet been estimated. You should first call the "estimate_baseline" method.')
        return baseline_path

    def save_threshold(self, folder_path: str, threshold_name: str) -> str:
        """ Stores persistently on disk the threshold.

        Args:
            folder_path     (:obj:`str`): Folder path where save the threshold.
            threshold_name  (:obj:`str`): Filename of the threshold file.

        Returns:
            :obj:`str`: The threshold filepath.
        """
        if self.threshold is not None:
            threshold_path = self.threshold.save(folder_path, threshold_name)
        else:
            raise Exception(f'Error: Threshold has not yet been estimated. You should first call the "estimate_threshold" method.')
        return threshold_path

    def load_baseline(self, folder_path: str, baseline_name: str) -> _baseline.BaselineClass:
        r""" Loads the baseline from disk into a BaselineClass object.

        Args:
            folder_path     (:obj:`str`): Folder path with the saved baseline.
            baseline_name   (:obj:`str`): Filename of the baseline folder.

        Returns:
            :class:`~driftlens._baseline.BaselineClass`: the loaded baseline.
        """
        baseline = _baseline.BaselineClass()

        baseline.load(folder_path=folder_path, baseline_name=baseline_name)

        self.baseline = baseline
        self.label_list = baseline.get_label_list()
        return baseline

    def set_baseline(self, baseline: _baseline.BaselineClass) -> None:
        """ Sets the baseline attribute with a BaselineClass object.

        Args:
            :class:`~driftlens._baseline.BaselineClass`: The baseline object to set.

        Returns:
            None
        """
        self.baseline = baseline
        return

    def set_threshold(self, threshold) -> None:
        """ Sets the threshold attribute with a ThresholdClass object.

        Args:
            :class:`~driftlens._threshold.ThresholdClass`: The threshold object to set.

        Returns:
            None
        """
        self.threshold = threshold
        return

    def random_sampling_threshold_estimation(self,
                                             label_list: list[int],
                                             E: np.ndarray,
                                             Y: np.ndarray,
                                             batch_n_pc: int,
                                             per_label_n_pc: int,
                                             window_size: int,
                                             n_samples: int,
                                             flag_shuffle: bool = True,
                                             flag_replacement: bool = True,
                                             proportional_flag: bool = False,
                                             proportions_dict=None,
                                             distribution_distance_metric: str ="frechet_drift_distance"
                                             ):
        """ Estimates the threshold using the random sampling algorithm.

        Args:
            label_list          (:obj:`list(int)`): List of class label ids used to train the model.
            E                   (:obj:`numpy.ndarray`): Embedding matrix of shape *(m, d)*, where *m* is the number of samples and *d* the embedding dimensionality.
            Y                   (:obj:`numpy.ndarray`): Vector of predicted labels of shape *(m, 1)*, where m is the number of samples.
            batch_n_pc          (:obj:`int`): Number of principal components to use for the per-batch.
            per_label_n_pc      (:obj:`int`): Number of principal components to use for the per-label.
            window_size         (:obj:`int`): Size of the window to use for the threshold estimation.
            n_samples           (:obj:`int`): Number of windows randomly sampled to use for the threshold estimation.
            flag_shuffle        (:obj:`bool`, `optional`): Flag to shuffle the samples before the threshold estimation. Default is True.
            flag_replacement    (:obj:`bool`, `optional`): Flag to sample with replacement the windows. Default is True.
            proportional_flag   (:obj:`bool`, `optional`): Flag to use the windows with proportional distribution between labels. Default is False.
            proportions_dict    (:obj:`dict`, `optional`): Dictionary with the proportions of the labels to use for the proportional sampling. Default is None.

        Returns:
            :obj:`tuple(numpy.ndarray, numpy.ndarray)`: Tuple with the per-batch distances sorted and the per-label distances.
        """

        threshold_algorithm = _threshold.RandomSamplingThresholdEstimator(label_list)
        # Execute the threshold estimation
        try:
            per_batch_distances_sorted, per_label_distances = threshold_algorithm.estimate_threshold(E, Y, self.baseline, window_size, n_samples, flag_shuffle=flag_shuffle, flag_replacement=flag_replacement, proportional_flag=proportional_flag, proportions_dict=proportions_dict, distribution_distance_metric=distribution_distance_metric)
        except Exception as e:
            raise Exception(f'Error in estimating the threshold: {e}')
        return per_batch_distances_sorted, per_label_distances

    def KFold_threshold_estimation(self,
                                   label_list: list[int],
                                   E: np.ndarray,
                                   Y: np.ndarray,
                                   batch_n_pc: int,
                                   per_label_n_pc: int,
                                   window_size: int,
                                   flag_shuffle: bool = True
                                   ):
        """ Estimates the threshold using the KFold algorithm (preliminary version of DriftLens).

        Args:
            label_list          (:obj:`list(int)`): List of class label ids used to train the model.
            E                   (:obj:`numpy.ndarray`): Embedding matrix of shape *(m, d)*, where *m* is the number of samples and *d* the embedding dimensionality.
            Y                   (:obj:`numpy.ndarray`): Vector of predicted labels of shape *(m, 1)*, where m is the number of samples.
            batch_n_pc          (:obj:`int`): Number of principal components to use for the per-batch.
            per_label_n_pc      (:obj:`int`): Number of principal components to use for the per-label.
            window_size         (:obj:`int`): Size of the window to use for the threshold estimation.
            flag_shuffle        (:obj:`bool`, `optional`): Flag to shuffle the samples before the threshold estimation. Default is True.

        Returns:
            :obj:`numpy.ndarray`: The estimated threshold.
        """
        threshold_algorithm = _threshold.KFoldThresholdEstimator(label_list)
        # Execute the threshold estimation
        try:
            self.threshold = threshold_algorithm.estimate_threshold(E, Y, batch_n_pc, per_label_n_pc, window_size, flag_shuffle=flag_shuffle)
        except Exception as e:
            raise Exception(f'Error in estimating the threshold: {e}')
        return self.threshold

    def repeated_KFold_threshold_estimation(self, label_list, E, Y, batch_n_pc, per_label_n_pc, window_size, repetitions, flag_shuffle=True):
        threshold_algorithm = _threshold.RepeatedKFoldThresholdEstimator(label_list)
        # Execute the threshold estimation
        try:
            self.threshold = threshold_algorithm.estimate_threshold(E, Y, batch_n_pc, per_label_n_pc, window_size, repetitions, flag_shuffle=flag_shuffle)
        except Exception as e:
            raise Exception(f'Error in estimating the threshold: {e}')
        return self.threshold

    def standard_threshold_estimation(self, label_list, E, Y, baseline, window_size, flag_shuffle=True):
        threshold_algorithm = _threshold.StandardThresholdEstimator(label_list)
        # Execute the threshold estimation
        try:
            self.threshold = threshold_algorithm.estimate_threshold(E, Y, baseline, window_size, flag_shuffle=flag_shuffle)
        except Exception as e:
            raise Exception(f'Error in estimating the threshold: {e}')
        return self.threshold

    def load_threshold(self, folder_path: str, threshold_name: str) -> _threshold.ThresholdClass:
        """ Loads the threshold from disk into a ThresholdClass object.

        Args:
            folder_path     (:obj:`str`): Folder path with the saved threshold
            threshold_name  (:obj:`str`): Filename of the threshold file.

        Returns:
            :class:`~driftlens._threshold.ThresholdClass`: The loaded threshold.
        """
        threshold = _threshold.ThresholdClass()

        threshold.load(folder_path=folder_path, threshold_name=threshold_name)

        self.threshold = threshold
        return threshold

    def compute_window_distribution_distances(self,
                                              E_w: np.ndarray,
                                              Y_w: np.ndarray,
                                              distribution_distance_metric: str = "frechet_drift_distance"
                                              ) -> dict:
        """ Computes the per-batch and per-label distribution distances for an embedding window.

        Args:
            E_w                             (:obj:`numpy.ndarray`): Embeddings of the window.
            Y_w                             (:obj:`numpy.ndarray`): Predicted labels of the window.
            distribution_distance_metric    (:obj:`str`, `optional`): The distribution distance metric to use. The Frechet Distance is used by default. Options are: ...

        Returns:
            a dictionary containing the per-batch (window_distribution_distances_dict[batch]) and the per-label
            (window_distribution_distances_dict[per-label][label]) distribution distances computed for the passed window
            with respect to the baseline.
        """
        if distribution_distance_metric == "frechet_drift_distance":
            window_distribution_distances_dict = self._compute_frechet_distribution_distances(self.label_list, self.baseline, E_w, Y_w)
        elif distribution_distance_metric == "mahalanobis_drift_distance":
            window_distribution_distances_dict = self._compute_mahalanobis_drift_distances(self.label_list, self.baseline, E_w, Y_w)
        elif distribution_distance_metric == "kullback_leibler_drift_divergence":
            window_distribution_distances_dict = self._compute_kullback_leibler_distribution_divergences(self.label_list, self.baseline, E_w, Y_w)
        elif distribution_distance_metric == "bhattacharyya_drift_distance":
            window_distribution_distances_dict = self._compute_bhattacharyya_distribution_distances(self.label_list, self.baseline, E_w, Y_w)
        elif distribution_distance_metric == "jensen_shannon_drift_divergence":
            window_distribution_distances_dict = self._compute_jensen_shannon_distribution_divergences(self.label_list, self.baseline, E_w, Y_w)
        else:
            return None
        return window_distribution_distances_dict

    def compute_window_list_distribution_distances(self,
                                                   E_w_list: List[np.ndarray],
                                                   Y_w_list: List[np.ndarray],
                                                   distribution_distance_metric: str = "frechet_drift_distance"
                                                   ) -> Tuple[List[dict], List[dict]]:
        """ Computes the per-batch and per-label distribution distances for each embedding window.

        Args:
            E_w_list                        (:obj:`list`(:obj:`numpy.ndarray`)`): List of embeddings of the windows.
            Y_w_list                        (:obj:`list`(:obj:`numpy.ndarray`)`): List of predicted labels of the windows.
            distribution_distance_metric    (:obj:`str`, `optional`): The distribution distance metric to use. Currently, only the Frechet Inception Distance is supported.

        Returns:
            :obj:`tuple`: A tuple containing a list of dictionaries containing the per-batch (window_distribution_distances_dict[batch]) and the per-label
            (window_distribution_distances_dict[per-label][label]) distribution distances computed for each input window
            with respect to the baseline.
        """
        window_distribution_list = []

        for window_id in range(len(E_w_list)):
            if distribution_distance_metric == "frechet_drift_distance":
                window_distribution_distances_dict = self._compute_frechet_distribution_distances(self.label_list,
                                                                                                  self.baseline,
                                                                                                  E_w_list[window_id],
                                                                                                  Y_w_list[window_id],
                                                                                                  window_id)
                window_distribution_list.append(window_distribution_distances_dict)
            elif distribution_distance_metric == "mahalanobis_drift_distance":
                window_distribution_distances_dict = self._compute_mahalanobis_drift_distances(self.label_list,
                                                                                                  self.baseline,
                                                                                                  E_w_list[window_id],
                                                                                                  Y_w_list[window_id],
                                                                                                  window_id)
                window_distribution_list.append(window_distribution_distances_dict)
            elif distribution_distance_metric == "kullback_leibler_drift_divergence":
                window_distribution_distances_dict = self._compute_kullback_leibler_distribution_divergences(self.label_list,
                                                                                                      self.baseline,
                                                                                                      E_w_list[window_id],
                                                                                                      Y_w_list[window_id],
                                                                                                      window_id)
                window_distribution_list.append(window_distribution_distances_dict)
            elif distribution_distance_metric == "bhattacharyya_drift_distance":
                window_distribution_distances_dict = self._compute_bhattacharyya_distribution_distances(self.label_list,
                                                                                                          self.baseline,
                                                                                                          E_w_list[window_id],
                                                                                                          Y_w_list[window_id],
                                                                                                          window_id)
                window_distribution_list.append(window_distribution_distances_dict)
            elif distribution_distance_metric == "jensen_shannon_drift_divergence":
                window_distribution_distances_dict = self._compute_jensen_shannon_distribution_divergences(self.label_list,
                                                                                                          self.baseline,
                                                                                                          E_w_list[window_id],
                                                                                                          Y_w_list[window_id],
                                                                                                          window_id)
                window_distribution_list.append(window_distribution_distances_dict)
            else:
                return None

        df_windows_distribution_distances = self.convert_distribution_distances_list_to_dataframe(window_distribution_list)
        return window_distribution_list, df_windows_distribution_distances


    # TODO tmp version
    def compute_drift_probability(self, window_distribution_list, threshold):

        # Initialize dicts
        per_batch_drift_probabilities = []
        per_label_drift_probabilities = {}
        for label in self.label_list:
            per_label_drift_probabilities[str(label)] = []

        # Compute drift probability for each window
        for window_dict in window_distribution_list:
            # Compute per_batch drift probability
            per_batch_distance = window_dict["per-batch"]
            per_batch_drift_probabilty = abs(0.5 - stats.norm.cdf(per_batch_distance, loc=threshold.get_batch_mean_distance(), scale=3*threshold.get_batch_std_distance())) * 100 / 0.5
            per_batch_drift_probabilities.append(per_batch_drift_probabilty)

            for label in self.label_list:
                per_label_distance = window_dict["per-label"][str(label)]
                per_label_drift_probabilty = abs(0.5 - stats.norm.cdf(per_label_distance, loc=threshold.get_mean_distance_by_label(str(label)),
                                             scale=3*threshold.get_std_distance_by_label(str(label)))) * 100 / 0.5
                per_label_drift_probabilities[str(label)].append(per_label_drift_probabilty)

        return {'per-batch': per_batch_drift_probabilities, 'per-label': per_label_drift_probabilities}

    @staticmethod
    def _compute_frechet_distribution_distances(label_list: List[int],
                                                baseline: _baseline.BaselineClass,
                                                E_w: np.ndarray,
                                                Y_w: np.ndarray,
                                                window_id: int = 0
                                                ) -> dict:
        """ Computes the frechet distribution distance (FDD) per-batch and per-label.

        Args:
            label_list  (:obj:`list(int)`): List of label ids.
            baseline    (:obj:`BaselineClass`): The baseline object.
            E_w         (:obj:`list`(:obj:`numpy.ndarray`)`): The embeddings of the current window.
            Y_w         (:obj:`list`(:obj:`numpy.ndarray`)`): The predicted labels of the current window.
            window_id   (:obj:`int`): The window id (default: 0).

        Returns:
            a dictionary containing the per-batch (window_distribution_distances_dict[batch]) and the per-label
        """
        window_distribution_distances_dict = {"window_id": window_id}

        mean_b_batch = baseline.get_batch_mean_vector()
        covariance_b_batch = baseline.get_batch_covariance_matrix()

        # Reduce the embedding dimensionality with PCA for the entire current window w
        E_w_reduced = baseline.get_batch_PCA_model().transform(E_w)

        mean_w_batch = fdd.get_mean(E_w_reduced)
        covariance_w_batch = fdd.get_covariance(E_w_reduced)

        distribution_distance_batch = fdd.frechet_distance(mean_b_batch,
                                                           mean_w_batch,
                                                           covariance_b_batch,
                                                           covariance_w_batch)

        window_distribution_distances_dict["per-batch"] = distribution_distance_batch
        window_distribution_distances_dict["per-label"] = {}

        for label in label_list:

            mean_b_l = baseline.get_mean_vector_by_label(label)
            covariance_b_l = baseline.get_covariance_matrix_by_label(label)

            # Select examples of of the current window w predicted with label l
            E_w_l_idxs = np.nonzero(Y_w == label)
            E_w_l = E_w[E_w_l_idxs]

            # Reduce the embedding dimensionality with PCA_l for current window w
            E_w_l_reduced = baseline.get_PCA_model_by_label(label).transform(E_w_l)

            # Estimate the mean vector and the covariance matrix for the label l in the current window w
            mean_w_l = fdd.get_mean(E_w_l_reduced)
            covariance_w_l = fdd.get_covariance(E_w_l_reduced)

            distribution_distance_l = fdd.frechet_distance(mean_b_l,
                                                           mean_w_l,
                                                           covariance_b_l,
                                                           covariance_w_l)

            window_distribution_distances_dict["per-label"][str(label)] = distribution_distance_l
        return window_distribution_distances_dict

    @staticmethod
    def _compute_mahalanobis_drift_distances(label_list: List[int],
                                             baseline: _baseline.BaselineClass,
                                             E_w: np.ndarray,
                                             Y_w: np.ndarray,
                                             window_id: int = 0
                                             ) -> dict:
        """ Computes the mahalanobis distribution distance per-batch and per-label.

        Args:
            label_list  (:obj:`list(int)`): List of label ids.
            baseline    (:obj:`BaselineClass`): The baseline object.
            E_w         (:obj:`list`(:obj:`numpy.ndarray`)`): The embeddings of the current window.
            Y_w         (:obj:`list`(:obj:`numpy.ndarray`)`): The predicted labels of the current window.
            window_id   (:obj:`int`): The window id (default: 0).

        Returns:
            a dictionary containing the per-batch (window_distribution_distances_dict[batch]) and the per-label
        """
        window_distribution_distances_dict = {"window_id": window_id}

        mean_b_batch = baseline.get_batch_mean_vector()
        covariance_b_batch = baseline.get_batch_covariance_matrix()

        # Reduce the embedding dimensionality with PCA for the entire current window w
        E_w_reduced = baseline.get_batch_PCA_model().transform(E_w)

        mean_w_batch = mdd.get_mean(E_w_reduced)

        distribution_distance_batch = mdd.mahalanobis_distance(mean_b_batch,
                                                               mean_w_batch,
                                                               covariance_b_batch)


        window_distribution_distances_dict["per-batch"] = distribution_distance_batch
        window_distribution_distances_dict["per-label"] = {}

        for label in label_list:
            mean_b_l = baseline.get_mean_vector_by_label(label)
            covariance_b_l = baseline.get_covariance_matrix_by_label(label)

            # Select examples of of the current window w predicted with label l
            E_w_l_idxs = np.nonzero(Y_w == label)
            E_w_l = E_w[E_w_l_idxs]

            # Reduce the embedding dimensionality with PCA_l for current window w
            E_w_l_reduced = baseline.get_PCA_model_by_label(label).transform(E_w_l)

            # Estimate the mean vector and the covariance matrix for the label l in the current window w
            mean_w_l = mdd.get_mean(E_w_l_reduced)

            distribution_distance_l = mdd.mahalanobis_distance(mean_b_l,
                                                           mean_w_l,
                                                           covariance_b_l)

            window_distribution_distances_dict["per-label"][str(label)] = distribution_distance_l

        return window_distribution_distances_dict

    @staticmethod
    def _compute_kullback_leibler_distribution_divergences(label_list: List[int],
                                                           baseline: _baseline.BaselineClass,
                                                           E_w: np.ndarray,
                                                           Y_w: np.ndarray,
                                                           window_id: int = 0
                                                           ) -> dict:
        """ Computes the frechet distribution distance (FID) per-batch and per-label.

        Args:
            label_list  (:obj:`list(int)`): List of label ids.
            baseline    (:obj:`BaselineClass`): The baseline object.
            E_w         (:obj:`list`(:obj:`numpy.ndarray`)`): The embeddings of the current window.
            Y_w         (:obj:`list`(:obj:`numpy.ndarray`)`): The predicted labels of the current window.
            window_id   (:obj:`int`): The window id (default: 0).

        Returns:
            a dictionary containing the per-batch (window_distribution_distances_dict[batch]) and the per-label
        """
        window_distribution_distances_dict = {"window_id": window_id}

        mean_b_batch = baseline.get_batch_mean_vector()
        covariance_b_batch = baseline.get_batch_covariance_matrix()

        # Reduce the embedding dimensionality with PCA for the entire current window w
        E_w_reduced = baseline.get_batch_PCA_model().transform(E_w)

        mean_w_batch = kldd.get_mean(E_w_reduced)
        covariance_w_batch = kldd.get_covariance(E_w_reduced)

        distribution_distance_batch = kldd.kl_divergence(mean_b_batch,
                                                           mean_w_batch,
                                                           covariance_b_batch,
                                                           covariance_w_batch)

        window_distribution_distances_dict["per-batch"] = distribution_distance_batch
        window_distribution_distances_dict["per-label"] = {}

        for label in label_list:
            mean_b_l = baseline.get_mean_vector_by_label(label)
            covariance_b_l = baseline.get_covariance_matrix_by_label(label)

            # Select examples of of the current window w predicted with label l
            E_w_l_idxs = np.nonzero(Y_w == label)
            E_w_l = E_w[E_w_l_idxs]

            # Reduce the embedding dimensionality with PCA_l for current window w
            E_w_l_reduced = baseline.get_PCA_model_by_label(label).transform(E_w_l)

            # Estimate the mean vector and the covariance matrix for the label l in the current window w
            mean_w_l = kldd.get_mean(E_w_l_reduced)
            covariance_w_l = kldd.get_covariance(E_w_l_reduced)

            distribution_distance_l = kldd.kl_divergence(mean_b_l,
                                                           mean_w_l,
                                                           covariance_b_l,
                                                           covariance_w_l)

            window_distribution_distances_dict["per-label"][str(label)] = distribution_distance_l
        return window_distribution_distances_dict

    @staticmethod
    def _compute_bhattacharyya_distribution_distances(label_list: List[int],
                                                      baseline: _baseline.BaselineClass,
                                                      E_w: np.ndarray,
                                                      Y_w: np.ndarray,
                                                      window_id: int = 0
                                                      ) -> dict:
        """ Computes the bhattacharyya distribution distance per-batch and per-label.

        Args:
            label_list  (:obj:`list(int)`): List of label ids.
            baseline    (:obj:`BaselineClass`): The baseline object.
            E_w         (:obj:`list`(:obj:`numpy.ndarray`)`): The embeddings of the current window.
            Y_w         (:obj:`list`(:obj:`numpy.ndarray`)`): The predicted labels of the current window.
            window_id   (:obj:`int`): The window id (default: 0).

        Returns:
            a dictionary containing the per-batch (window_distribution_distances_dict[batch]) and the per-label
        """
        window_distribution_distances_dict = {"window_id": window_id}

        mean_b_batch = baseline.get_batch_mean_vector()
        covariance_b_batch = baseline.get_batch_covariance_matrix()

        # Reduce the embedding dimensionality with PCA for the entire current window w
        E_w_reduced = baseline.get_batch_PCA_model().transform(E_w)

        mean_w_batch = fdd.get_mean(E_w_reduced)
        covariance_w_batch = bdd.get_covariance(E_w_reduced)

        distribution_distance_batch = bdd.bhattacharyya_distance(mean_b_batch,
                                                                   mean_w_batch,
                                                                   covariance_b_batch,
                                                                   covariance_w_batch)

        window_distribution_distances_dict["per-batch"] = distribution_distance_batch
        window_distribution_distances_dict["per-label"] = {}

        for label in label_list:
            mean_b_l = baseline.get_mean_vector_by_label(label)
            covariance_b_l = baseline.get_covariance_matrix_by_label(label)

            # Select examples of of the current window w predicted with label l
            E_w_l_idxs = np.nonzero(Y_w == label)
            E_w_l = E_w[E_w_l_idxs]

            # Reduce the embedding dimensionality with PCA_l for current window w
            E_w_l_reduced = baseline.get_PCA_model_by_label(label).transform(E_w_l)

            # Estimate the mean vector and the covariance matrix for the label l in the current window w
            mean_w_l = bdd.get_mean(E_w_l_reduced)
            covariance_w_l = bdd.get_covariance(E_w_l_reduced)

            distribution_distance_l = bdd.bhattacharyya_distance(mean_b_l,
                                                                   mean_w_l,
                                                                   covariance_b_l,
                                                                   covariance_w_l)

            window_distribution_distances_dict["per-label"][str(label)] = distribution_distance_l
        return window_distribution_distances_dict

    @staticmethod
    def _compute_jensen_shannon_distribution_divergences(label_list: List[int],
                                                         baseline: _baseline.BaselineClass,
                                                         E_w: np.ndarray,
                                                         Y_w: np.ndarray,
                                                         window_id: int = 0
                                                         ) -> dict:
        """ Computes the jensen shannon distribution distance per-batch and per-label.

        Args:
            label_list  (:obj:`list(int)`): List of label ids.
            baseline    (:obj:`BaselineClass`): The baseline object.
            E_w         (:obj:`list`(:obj:`numpy.ndarray`)`): The embeddings of the current window.
            Y_w         (:obj:`list`(:obj:`numpy.ndarray`)`): The predicted labels of the current window.
            window_id   (:obj:`int`): The window id (default: 0).

        Returns:
            a dictionary containing the per-batch (window_distribution_distances_dict[batch]) and the per-label
        """
        window_distribution_distances_dict = {"window_id": window_id}

        mean_b_batch = baseline.get_batch_mean_vector()
        covariance_b_batch = baseline.get_batch_covariance_matrix()

        # Reduce the embedding dimensionality with PCA for the entire current window w
        E_w_reduced = baseline.get_batch_PCA_model().transform(E_w)

        mean_w_batch = jsdd.get_mean(E_w_reduced)
        covariance_w_batch = jsdd.get_covariance(E_w_reduced)

        distribution_distance_batch = jsdd.jensen_shannon_divergence(mean_b_batch,
                                                                 mean_w_batch,
                                                                 covariance_b_batch,
                                                                 covariance_w_batch)

        window_distribution_distances_dict["per-batch"] = distribution_distance_batch
        window_distribution_distances_dict["per-label"] = {}

        for label in label_list:
            mean_b_l = baseline.get_mean_vector_by_label(label)
            covariance_b_l = baseline.get_covariance_matrix_by_label(label)

            # Select examples of of the current window w predicted with label l
            E_w_l_idxs = np.nonzero(Y_w == label)
            E_w_l = E_w[E_w_l_idxs]

            # Reduce the embedding dimensionality with PCA_l for current window w
            E_w_l_reduced = baseline.get_PCA_model_by_label(label).transform(E_w_l)

            # Estimate the mean vector and the covariance matrix for the label l in the current window w
            mean_w_l = jsdd.get_mean(E_w_l_reduced)
            covariance_w_l = jsdd.get_covariance(E_w_l_reduced)

            distribution_distance_l = jsdd.jensen_shannon_divergence(mean_b_l,
                                                                     mean_w_l,
                                                                     covariance_b_l,
                                                                     covariance_w_l)

            window_distribution_distances_dict["per-label"][str(label)] = distribution_distance_l
        return window_distribution_distances_dict

    @staticmethod
    def convert_distribution_distances_list_to_dataframe(distribution_distances_list: dict) -> pd.DataFrame:
        """ Converts the list of distribution distances to a pandas DataFrame.
        
        Args:   
            distribution_distances_list (:obj:`list(dict)`): A list of dictionaries containing the distribution distances.

        Returns:
            :obj:`pd.DataFrame`: A pandas DataFrame containing the distribution distances.
        """
        if type(distribution_distances_list) is dict:
            distribution_distances_list = [distribution_distances_list]

        dict_list = []
        for distribution_distances_dict in distribution_distances_list:
            d = {}
            d["window_id"] = distribution_distances_dict["window_id"]
            d["batch_distance"] = distribution_distances_dict["per-batch"]
            for label, distance in distribution_distances_dict["per-label"].items():
                d["label_{}_distance".format(label)] = distance

            dict_list.append(d)
        return pd.DataFrame(dict_list)


class DriftLensVisualizer:
    """ Class to visualize the drift detection monitor results. """
    def __init__(self):
        return

    @staticmethod
    def _parse_distribution_distances(label_list, windows_distribution_distances):
        """ Parse the distribution distances to per-label and per-batch distances.
        Args:
            label_list (:obj:`list(int)`): list of label ids.
            windows_distribution_distances (:obj:`list(dict)`): list of distribution distances.
        Returns:
            per_label_distribution_distances (dict): dictionary with per-label distribution distances.
            per_batch_distribution_distances (list): list of per-batch distribution distances.
         """
        per_label_distribution_distances = {}
        per_batch_distribution_distances = []

        for l in label_list:
            per_label_distribution_distances[str(l)] = []

        for window_distribution_distances in windows_distribution_distances:
            per_batch_distribution_distances.append(window_distribution_distances["per-batch"])
            for l in label_list:
                per_label_distribution_distances[str(l)].append(window_distribution_distances["per-label"][str(l)])
        return per_label_distribution_distances, per_batch_distribution_distances

    @staticmethod
    def plot_per_label_drift_monitor(window_distribution_list, label_names=None, plt_title=None, plt_xlabel_name=None, plt_ylabel_name=None, ylim_top=15,
                                     flag_save=False, folder_path=None, filename=None, format='eps'):

        label_list = window_distribution_list[0]["per-label"].keys()

        per_label_distribution_distances, per_batch_distribution_distances = DriftLensVisualizer()._parse_distribution_distances(label_list, window_distribution_list)
        windows_distribution_distances = per_label_distribution_distances

        if label_names is None:
            label_names = []
            for l in label_list:
                label_names.append("Label {}".format(l))
        else:
            if len(label_list) != len(label_names):
                raise Exception("Error")

        x_axis = range(len(window_distribution_list))

        for l in label_list:
            p = plt.plot(x_axis, utils.clear_complex_numbers(windows_distribution_distances[str(l)]))

        for l in label_list:
            plt.scatter(x_axis, utils.clear_complex_numbers(windows_distribution_distances[str(l)]))

        if plt_title is not None:
            plt.title(plt_title)

        plt.xticks(x_axis)

        if plt_xlabel_name is None:
            plt.xlabel("Windows", fontsize=12)
        else:
            plt.xlabel(plt_xlabel_name, fontsize=12)

        if plt_ylabel_name is None:
            plt.ylabel("FID Score", fontsize=12)
        else:
            plt.ylabel(plt_ylabel_name, fontsize=12)

        plt.legend(label_names, loc='upper left')
        plt.ylim(top=ylim_top)
        plt.tight_layout()
        plt.grid(True, linestyle="dashed", alpha=0.5)

        if flag_save:
            if folder_path is None:
                folder_path = ''

            if filename is None:
                filename = 'drift_lens_per_label_monitor'

            filename = filename + "." + format
            plt.savefig(os.path.join(folder_path, filename), format=format, dpi=1800)

        plt.show()
        return

    @staticmethod
    def plot_per_batch_drift_monitor(window_distribution_list, plt_title=None, plt_xlabel_name=None, plt_ylabel_name=None, ylim_top=15,
                                     flag_save=False, folder_path=None, filename=None, format='eps'):
        label_list = window_distribution_list[0]["per-label"].keys()

        per_label_distribution_distances, per_batch_distribution_distances = DriftLensVisualizer()._parse_distribution_distances(
            label_list, window_distribution_list)
        windows_distribution_distances = per_label_distribution_distances


        x_axis = range(len(window_distribution_list))

        p = plt.plot(x_axis, utils.clear_complex_numbers(per_batch_distribution_distances))
        plt.scatter(x_axis, utils.clear_complex_numbers(per_batch_distribution_distances))

        print(utils.clear_complex_numbers(per_batch_distribution_distances))

        if plt_title is not None:
            plt.title(plt_title)

        plt.xticks(x_axis)

        if plt_xlabel_name is None:
            plt.xlabel("Windows", fontsize=12)
        else:
            plt.xlabel(plt_xlabel_name, fontsize=12)

        if plt_ylabel_name is None:
            plt.ylabel("FID Score", fontsize=12)
        else:
            plt.ylabel(plt_ylabel_name, fontsize=12)

        plt.legend(["per-batch"], loc='upper left')
        plt.ylim(top=ylim_top)
        plt.tight_layout()
        plt.grid(True, linestyle="dashed", alpha=0.5)

        if flag_save:
            if folder_path is None:
                folder_path = ''

            if filename is None:
                filename = 'drift_lens_per_batch_monitor'

            filename = filename + "." + format
            plt.savefig(os.path.join(folder_path, filename), format=format, dpi=1800)

        plt.show()
        return

    # TODO sistemare codice
    def plot_per_label_monitor_with_threshold(self, label_names=None, ylim_top=15):
        if label_names is None:
            label_names = []
            for l in self.training_label_list:
                label_names.append("Label {}".format(l))

        x_axis = range(len(self.windows_distribution_distances))

        for l in self.training_label_list:
            p = plt.plot(x_axis, utils.clear_complex_numbers(self.per_label_distribution_distances[str(l)]))

        for l in self.training_label_list:
            plt.scatter(x_axis, utils.clear_complex_numbers(self.per_label_distribution_distances[str(l)]))
            # plt.scatter(x_axis, self.per_label_distribution_distances[str(l)], c=p[-1].get_color())

        # names.append("Drift")

        # plt.title("Gradual Drift - Win 2000")
        plt.xticks(x_axis)
        plt.xlabel("Windows", fontsize=12)
        plt.ylabel("FID Score", fontsize=12)
        plt.legend(label_names, loc='upper left')
        # ax = plt.gca()
        # leg = ax.get_legend()
        # leg.legendHandles[-1].set_color('black')
        plt.ylim(top=ylim_top)
        # for line in ts_drift:
        #    plt.axvline(x=line, color='grey', alpha=1, linewidth=0.75)
        # if len(ts_drift) > 0:
        #    plt.text(3.1, 0.9, 'Drift', rotation=90, alpha=1, color='grey', va='top')
        plt.tight_layout()
        plt.grid(True, linestyle="dashed", alpha=0.5)

        # plt.savefig('tmp.eps', format='eps', dpi=1800)
        plt.show()

        return


