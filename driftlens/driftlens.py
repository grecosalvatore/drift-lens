import os.path

from driftlens import _frechet_drift_distance as fdd
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
        baseline (:obj:`BaselineClass`): BaselineClass object.
        threshold (:obj:`ThresholdClass`): ThresholdClass object.
        label_list (:obj:`list(str)`): List of class labels.
        batch_n_pc (:obj:`int`): Number of principal components to use for the per-batch.
        per_label_n_pc (:obj:`int`): Number of principal components to use for the per-label.
        baseline_algorithms (:obj:`dict`): Dictionary of possible baseline algorithms.
        threshold_estimators (:obj:`dict`): Dictionary of possible threshold estimators.
    """
    def __init__(self, label_list=None):

        self.baseline = None  # BaselineClass object
        self.threshold = None  # ThresholdClass object

        self.label_list = label_list  # List of class labels
        self.batch_n_pc = None  # Number of principal components to use for the per-batch
        self.per_label_n_pc = None  # Number of principal components to use for the per-label

        self.baseline_algorithms = {"StandardBaselineEstimator": "Description"}
        self.threshold_estimators = {"KFoldThresholdEstimator": "Description"}

    def estimate_baseline(self, E, Y,  label_list, batch_n_pc, per_label_n_pc, baseline_algorithm="StandardBaselineEstimator") -> _baseline.BaselineClass:
        r""" Estimates the baseline.

        Args:
            label_list (:obj:`list(str)`): List of class labels used to train the model.
            batch_n_pc (:obj:`int`): Number of principal components to use for the per-batch.
            per_label_n_pc (:obj:`int`): Number of principal components to use for the per-label.
            E (:obj:`numpy.ndarray`): Embedding matrix of shape (m, d), where m is the number of samples and d the embedding dimensionality.
            Y (:obj:`numpy.ndarray`): Vector of predicted labels of shape (m, 1), where m is the number of samples.
            baseline_algorithm (:obj:`str`, `optional`): Baseline estimation algorithm to use. Possible values are: "StandardBaselineEstimator". If not provided, the default value is "StandardBaselineEstimator".

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

    def save_baseline(self, folder_path, baseline_name) -> str:
        """ Stores persistently on disk the baseline.

        Args:
            folder_path (:obj:`str`): Folder path where save the baseline.
            baseline_name (:obj:`str`): Filename of the baseline folder.

        Returns:
            :obj:`str`: Baseline folder path.
        """
        if self.baseline is not None:
            baseline_path = self.baseline.save(folder_path, baseline_name)
        else:
            raise Exception(f'Error: Baseline has not yet been estimated. You should first call the "estimate_baseline" method.')
        return baseline_path

    def save_threshold(self, folder_path, threshold_name):
        """ Stores persistently on disk the threshold.

        Args:
            folder_path (str): Folder path where save the threshold.
            threshold_name (str): Filename of the threshold file.
        Returns:
            (str): threshold filepath.
        """
        if self.threshold is not None:
            threshold_path = self.threshold.save(folder_path, threshold_name)
        else:
            raise Exception(f'Error: Threshold has not yet been estimated. You should first call the "estimate_threshold" method.')
        return threshold_path

    def load_baseline(self, folder_path, baseline_name) -> _baseline.BaselineClass:
        r""" Loads the baseline from disk into a BaselineClass object.

        Args:
            folder_path (:obj:`str`): Folder path with the saved baseline.
            baseline_name (:obj:`str`): Filename of the baseline folder.
        Returns:
            :class:`~driftlens._baseline.BaselineClass`: the loaded baseline.
        """
        baseline = _baseline.BaselineClass()

        baseline.load(folder_path=folder_path, baseline_name=baseline_name)

        self.baseline = baseline
        self.label_list = baseline.get_label_list()
        return baseline

    def set_baseline(self, baseline) -> None:
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

    def random_sampling_threshold_estimation(self, label_list, E, Y, batch_n_pc, per_label_n_pc, window_size, n_samples, flag_shuffle=True, flag_replacement=True, proportional_flag=False, proportions_dict=None):
        threshold_algorithm = _threshold.RandomSamplingThresholdEstimator(label_list)
        # Execute the threshold estimation
        try:
            per_batch_distances_sorted, per_label_distances = threshold_algorithm.estimate_threshold(E, Y, self.baseline, window_size, n_samples, flag_shuffle=flag_shuffle, flag_replacement=flag_replacement, proportional_flag=proportional_flag, proportions_dict=proportions_dict)
        except Exception as e:
            raise Exception(f'Error in estimating the threshold: {e}')
        return per_batch_distances_sorted, per_label_distances

    def KFold_threshold_estimation(self, label_list, E, Y, batch_n_pc, per_label_n_pc, window_size, flag_shuffle=True):
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

    def load_threshold(self, folder_path, threshold_name) -> _threshold.ThresholdClass:
        """ Loads the threshold from disk into a ThresholdClass object.

        Args:
            folder_path (:obj:`str`): Folder path with the saved threshold
            threshold_name (:obj:`str`): Filename of the threshold file.

        Returns:
            :class:`~driftlens._threshold.ThresholdClass`: The loaded threshold.
        """
        threshold = _threshold.ThresholdClass()

        threshold.load(folder_path=folder_path, threshold_name=threshold_name)

        self.threshold = threshold
        return threshold

    def compute_window_distribution_distances(self, E_w, Y_w, distribution_distance_metric="frechet_inception_distance"):
        """ Computes the per-batch and per-label distribution distances for an embedding window.

        Args:
            E_w:
            Y_w:
            distribution_distance_metric (str):

        Returns:
            a dictionary containing the per-batch (window_distribution_distances_dict[batch]) and the per-label
            (window_distribution_distances_dict[per-label][label]) distribution distances computed for the passed window
            with respect to the baseline.
        """
        if distribution_distance_metric == "frechet_inception_distance":
            window_distribution_distances_dict = self._compute_frechet_distribution_distances(self.label_list, self.baseline, E_w, Y_w)
        else:
            return None
        return window_distribution_distances_dict

    def compute_window_list_distribution_distances(self, E_w_list, Y_w_list, distribution_distance_metric="frechet_inception_distance"):
        """ Computes the per-batch and per-label distribution distances for each embedding window.
        Args:
            E_w_list:
            Y_w_list:
            distribution_distance_metric (str):

        Returns:
            a list of dictionaries containing the per-batch (window_distribution_distances_dict[batch]) and the per-label
            (window_distribution_distances_dict[per-label][label]) distribution distances computed for each input window
            with respect to the baseline.
        """
        window_distribution_list = []

        for window_id in range(len(E_w_list)):
            if distribution_distance_metric == "frechet_inception_distance":
                window_distribution_distances_dict = self._compute_frechet_distribution_distances(self.label_list,
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
    def _compute_frechet_distribution_distances(label_list, baseline,  E_w, Y_w, window_id=0):
        """ Computes the frechet distribution distance (FID) per-batch and per-label.
        Args:
            label_list (list):
            baseline
            E_w:
            Y_w:
            window_id:
        Returns:


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
    def convert_distribution_distances_list_to_dataframe(distribution_distances_list):

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
            label_list (list): list of labels.
            windows_distribution_distances (list): list of distribution distances.
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


