from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from driftlens import driftlens
from tqdm import tqdm
import numpy as np
import json
import os
from tqdm import tqdm


class ThresholdClass:
    """ Threshold Class: it contains all the attributes and methods of the threshold. """
    def __init__(self):
        self.label_list = None  # List of labels used to train the model
        self.batch_n_pc = None  # Number of principal components to reduce the embedding for the entire batch drift
        self.per_label_n_pc = None  # number of principal components to reduce embedding for the per-label drift
        self.window_size = None  # Window size that will be used in the online phase

        self.mean_distances_dict = {}
        self.std_distances_dict = {}

        self.mean_distances_dict["per-label"] = {}
        self.std_distances_dict["per-label"] = {}

        self.mean_distances_dict["batch"] = None
        self.std_distances_dict["batch"] = None

        self.distribution_distances_list = None

        self.description = ""
        self.threshold_method_name = ""
        return

    def fit(self, batch_mean, batch_std, per_label_mean_dict, per_label_std_dict, label_list, batch_n_pc, per_label_n_pc, window_size, distribution_distances_list,  threshold_method_name=""):
        """ Fits the threshold attributes.
        Args:
            batch_mean:

        Returns:
            None
        """

        self.mean_distances_dict["batch"] = batch_mean
        self.std_distances_dict["batch"] = batch_std

        self.mean_distances_dict["per-label"] = per_label_mean_dict
        self.std_distances_dict["per-label"] = per_label_std_dict

        self.distribution_distances_list = distribution_distances_list

        self.label_list = label_list
        self.batch_n_pc = batch_n_pc
        self.per_label_n_pc = per_label_n_pc
        self.window_size = window_size

        self.threshold_method_name = threshold_method_name
        return

    def _fit_from_dict(self, threshold_dict):
        """ Fits the threshold class attributes from a dictionary. """
        self.__dict__.update(threshold_dict)
        return

    def save(self, folderpath, threshold_name):
        """ Saves persistently on disk the threshold.
        Args:
            folderpath (str): Folder path where save the threshold.
            threshold_name (str): Filename of the threshold file.
        Returns:
            (str): Threshold filepath.
        """

        # Serialize the self object to json
        experiment_json_str = json.dumps(self, default=lambda x: getattr(x, '__dict__', str(x)))

        experiment_json_dict = json.loads(experiment_json_str)

        # Save the json to file
        with open(os.path.join(folderpath, "{}.json".format(threshold_name)), 'w') as fp:
            json.dump(experiment_json_dict, fp)
        return

    def load(self, folderpath, threshold_name):
        """ Loads the threshold from folder.
        Args:
            folderpath: (str) folderpath containing the threshold json.
            threshold_name: (str) name of the threshold json file.
        """
        if threshold_name.endswith(".json"):
            threshold_filepath = os.path.join(folderpath, threshold_name)
        else:
            threshold_filepath = '{}.json'.format(os.path.join(folderpath, threshold_name))

        with open(threshold_filepath) as json_file:
            json_dict = json.load(json_file)

        if json_dict is None:
            raise Exception(f'Error: impossible to parse threshold json.')
        else:
            try:
                self._fit_from_dict(json_dict)
            except Exception as e:
                raise Exception(f'Error in deserializing the  threshold json: {e}')

        return

    def set_description(self, description):
        """ Sets the 'description' attribute of the threshold.
        Args:
            (str): Description of the threshold.
        """
        self.description = description
        return

    def get_description(self):
        """ Gets the 'description' attribute of the threshold.
         Returns:
             (str): Description of the threshold.
        """
        return self.description

    def get_label_list(self):
        """ Gets the 'label_list' attribute of the threshold. It contains the list of labels.
        Returns:
            (list): List of labels used to train the model.
        """
        return self.label_list

    def get_batch_mean_distance(self):
        """ Gets the 'mean_distances_dict['batch']' attribute of the threshold. It contains the mean of the distribution distances computed on the threshold dataset over the entire batch. """
        return self.mean_distances_dict["batch"]

    def get_batch_std_distance(self):
        """ Gets the 'std_distances_dict['batch']' attribute of the threshold. It contains the standard deviation of the distribution distances computed on the threshold dataset over the entire batch. """
        return self.std_distances_dict["batch"]

    def get_mean_distance_by_label(self, label):
        """ Gets the 'mean_distances_dict['per-label'][label]' attribute of the threshold. It contains the per-label mean of the distribution distances computed on the threshold dataset for samples associated to label as parameter. """
        return self.mean_distances_dict["per-label"][str(label)]

    def get_std_distance_by_label(self, label):
        """ Gets the 'std_distances_dict['per-label'][label]' attribute of the threshold. It contains the per-label standard deviation of the distribution distances computed on the threshold dataset for samples associated to label as parameter. """
        return self.std_distances_dict["per-label"][str(label)]

    def get_per_label_mean_distances(self):
        """ Gets the 'mean_distances_dict['per-label']' dictionary attribute of the threshold. It contains the per-label mean of the distribution distances computed on the threshold dataset for samples associated to each label (for each label). """
        return self.mean_distances_dict["per-label"]

    def get_per_label_std_distances(self):
        """ Gets the 'std_distances_dict['per-label']' dictionary attribute of the threshold. It contains the per-label standard deviation of the distribution distances computed on the threshold dataset for samples associated to each label (for each label). """
        return self.std_distances_dict["per-label"]


class ThresholdEstimatorMethod(ABC):
    """ Abstract Baseline Estimator Method class. """

    def __init__(self, label_list, threshold_method_name):
        self.label_list = label_list
        self.threshold_method_name = threshold_method_name
        return

    @abstractmethod
    def estimate_threshold(self, *args) -> ThresholdClass:
        """ Abstract method: Estimates the Threshold and returns a ThresholdClass object. """
        pass


class KFoldThresholdEstimator(ThresholdEstimatorMethod):
    """ KFold Threshold Estimator Class: Implementation of the ThresholdEstimatorMethod Abstract Class. """
    def __init__(self, label_list):
        ThresholdEstimatorMethod.__init__(self, label_list, threshold_method_name="KFoldThresholdEstimator")
        return

    def estimate_threshold(self, E, Y, batch_n_pc, per_label_n_pc, window_size, flag_shuffle=True):
        """ Implementation of the 'estimate_threshold' Abstract method: Estimates the Threshold and returns a ThresholdClass object. """

        if window_size > E.shape[0] * 2:
            # Error
            print("Error")

        # Compute the number of folds K as: number of samples/window size
        K = round(E.shape[0] / window_size)

        E_selected = E[:window_size*K]
        Y_selected = Y[:window_size*K]

        distribution_distances_list = self._kfold_thredhold_estimation(E_selected, Y_selected, K, batch_n_pc, per_label_n_pc, flag_shuffle, start_window_id=0)

        batch_mean = np.mean(distribution_distances_list["batch"])
        batch_std = np.std(distribution_distances_list["batch"])
        per_label_mean = {}
        per_label_std = {}

        for label in self.label_list:
            per_label_mean[str(label)] = np.mean(distribution_distances_list["per-label"][str(label)])
            per_label_std[str(label)] = np.std(distribution_distances_list["per-label"][str(label)])

        threshold = ThresholdClass()

        threshold.fit(batch_mean, batch_std, per_label_mean, per_label_std, self.label_list, batch_n_pc, per_label_n_pc,
                      window_size, distribution_distances_list, self.threshold_method_name)

        return threshold

    def _kfold_thredhold_estimation(self,E, Y, K, batch_n_pc, per_label_n_pc, flag_shuffle, start_window_id=0):
        window_id = start_window_id

        distribution_distances_list = {"batch": [], "per-label":{}}
        distribution_distances_list["per-label"] = {str(l): [] for l in self.label_list}

        folds = StratifiedKFold(n_splits=K, shuffle=flag_shuffle, random_state=0)

        for approximated_baseline_idxs, simulated_new_window_idx in folds.split(E, Y):
            E_b, E_w = E[approximated_baseline_idxs], E[simulated_new_window_idx]
            Y_b, Y_w = Y[approximated_baseline_idxs], Y[simulated_new_window_idx]

            # Estimate an approximated baseline with k-1 folds
            approximated_baseline = self._baseline.StandardBaselineEstimator(self.label_list, batch_n_pc,
                                                                        per_label_n_pc).estimate_baseline(E_b, Y_b)

            window_distribution_distances_dict = driftlens.DriftLens()._compute_frechet_distribution_distances(
                self.label_list, approximated_baseline, E_w, Y_w, window_id)

            distribution_distances_list["batch"].append(window_distribution_distances_dict["batch"])

            for l in self.label_list:
                distribution_distances_list["per-label"][str(l)].append(
                    window_distribution_distances_dict["per-label"][str(l)])
            window_id += 1

        return distribution_distances_list


class RepeatedKFoldThresholdEstimator(KFoldThresholdEstimator):
    """ Repeated KFold Threshold Estimator Class: Implementation of the ThresholdEstimatorMethod Abstract Class. """
    def __init__(self, label_list):
        ThresholdEstimatorMethod.__init__(self, label_list, threshold_method_name="RepeatedKFoldThresholdEstimator")
        return

    def estimate_threshold(self, E, Y, batch_n_pc, per_label_n_pc, window_size, repetitions=2, flag_shuffle=True):
        """ Implementation of the 'estimate_threshold' Abstract method: Estimates the Threshold and returns a ThresholdClass object. """

        if window_size > E.shape[0] * 2:
            # Error
            print("Error")

        # Compute the number of folds K as: number of samples/window size
        K = round(E.shape[0] / window_size)

        E_selected = E[:window_size*K]
        Y_selected = Y[:window_size*K]

        distribution_distances_list = {"batch": [], "per-label": {}}
        distribution_distances_list["per-label"] = {str(l): [] for l in self.label_list}

        for r in tqdm(range(repetitions)):

            partial_distribution_distances_list = self._kfold_thredhold_estimation(E_selected, Y_selected, K, batch_n_pc, per_label_n_pc, flag_shuffle=True, start_window_id=r)

            distribution_distances_list["batch"] = distribution_distances_list["batch"] + partial_distribution_distances_list["batch"]
            for label in self.label_list:
                distribution_distances_list["per-label"][str(label)] = distribution_distances_list["per-label"][str(label)] + partial_distribution_distances_list["per-label"][str(label)]

        batch_mean = np.mean(distribution_distances_list["batch"])
        batch_std = np.std(distribution_distances_list["batch"])
        per_label_mean = {}
        per_label_std = {}

        for label in self.label_list:
            per_label_mean[str(label)] = np.mean(distribution_distances_list["per-label"][str(label)])
            per_label_std[str(label)] = np.std(distribution_distances_list["per-label"][str(label)])

        threshold = ThresholdClass()

        threshold.fit(batch_mean, batch_std, per_label_mean, per_label_std, self.label_list, batch_n_pc, per_label_n_pc,
                      window_size, distribution_distances_list, self.threshold_method_name)

        return threshold


class StandardThresholdEstimator(ThresholdEstimatorMethod):
    """ Standard Threshold Estimator Class: Implementation of the ThresholdEstimatorMethod Abstract Class.

    The StandardThresholdEstimator estimates a threshold by dividing a new unseen dataset fixed windows, and by computing
    the distribution distances of each windows with a respect of a given baseline.

    This method should be used when a different dataset from the baseline dataset is available.
    """
    def __init__(self, label_list):
        ThresholdEstimatorMethod.__init__(self, label_list, threshold_method_name="StandardThresholdEstimator")
        return

    def estimate_threshold(self, E, Y, baseline, window_size, flag_shuffle=True):
        """ Implementation of the 'estimate_threshold' Abstract method: Estimates the Threshold and returns a ThresholdClass object. """

        # Number of principal components for both batch and per-label are extracted from the baseline
        batch_n_pc = baseline.batch_n_pc
        per_label_n_pc = baseline.per_label_n_pc

        # Check if the number of samples is a multiple of window_size
        if E.shape[0] % window_size == 0:
            print("Warning: Number of samples is not an exact multiple of window size. {} samples are discarded.".format(E.shape[0] // window_size))

        # If flag_shuffle is True, then shuffle embedding vectors and label accordingly
        if flag_shuffle:
            p = np.random.permutation(len(E))
            E = E[p]
            Y = Y[p]

        # The number of windows is set to the number of possible full size window_size windows
        n_windows = E.shape[0] // window_size

        distribution_distances_list = {"batch": [], "per-label": {}}
        distribution_distances_list["per-label"] = {str(l): [] for l in self.label_list}

        for i in range(n_windows):

            window_id = i

            E_w = E[i*window_size:(i*window_size)+window_size]
            Y_w = Y[i*window_size:(i*window_size)+window_size]

            window_distribution_distances_dict = driftlens.DriftLens()._compute_frechet_distribution_distances(self.label_list, baseline, E_w, Y_w, window_id)

            distribution_distances_list["batch"].append(window_distribution_distances_dict["batch"])

            for l in self.label_list:
                distribution_distances_list["per-label"][str(l)].append(window_distribution_distances_dict["per-label"][str(l)])

        batch_mean = np.mean(distribution_distances_list["batch"])
        batch_std = np.std(distribution_distances_list["batch"])
        per_label_mean = {}
        per_label_std = {}

        for label in self.label_list:
            per_label_mean[str(label)] = np.mean(distribution_distances_list["per-label"][str(label)])
            per_label_std[str(label)] = np.std(distribution_distances_list["per-label"][str(label)])

        threshold = ThresholdClass()

        threshold.fit(batch_mean, batch_std, per_label_mean, per_label_std, self.label_list, batch_n_pc, per_label_n_pc, window_size, distribution_distances_list, self.threshold_method_name)

        return threshold


class RandomSamplingThresholdEstimator(ThresholdEstimatorMethod):
    """ Random Sampling Threshold Estimator Class: Implementation of the ThresholdEstimatorMethod Abstract Class. """

    def __init__(self, label_list):
        ThresholdEstimatorMethod.__init__(self, label_list, threshold_method_name="RandomSamplingThresholdEstimator")
        return

    def estimate_threshold(self, E, Y, baseline, window_size, n_samples, flag_replacement=True, flag_shuffle=True, proportional_flag=False, proportions_dict=None, distribution_distance_metric="frechet_inception_distance"):
        """ Implementation of the 'estimate_threshold' Abstract method: Estimates the Threshold and returns a ThresholdClass object. """

        print(f"Estimating threshold using {distribution_distance_metric} metric")

        per_batch_distances = []
        per_label_distances = {label: [] for label in self.label_list}

        print("Threshold Estimation")
        for i in tqdm(range(n_samples)):

            if proportional_flag == False:

                E_windows, Y_predicted_windows, Y_original_windows = self._balanced_sampling(self.label_list,
                                                                                             E,
                                                                                             Y,
                                                                                             Y,
                                                                                             window_size,
                                                                                             1,
                                                                                             flag_replacement)
            else:
                E_windows, Y_predicted_windows, Y_original_windows = self._proportional_sampling(self.label_list,
                                                                                                 E,
                                                                                                 Y,
                                                                                                 Y,
                                                                                                 window_size,
                                                                                                 1,
                                                                                                 flag_replacement,
                                                                                                 proportions_dict)

            E_windows[0], Y_predicted_windows[0], Y_original_windows[0] = self._shuffle_dataset(E_windows[0],
                                                                                                Y_predicted_windows[0],
                                                                                                Y_original_windows[0])

            dl_th = driftlens.DriftLens(self.label_list)
            dl_th.set_baseline(baseline)
            distribution_distances = dl_th.compute_window_list_distribution_distances(E_windows, Y_predicted_windows, distribution_distance_metric=distribution_distance_metric)


            per_batch_distances.append(distribution_distances[0][0]["per-batch"])
            for l in self.label_list:
                per_label_distances[l].append(distribution_distances[0][0]["per-label"][str(l)])


        per_batch_distances_arr = np.array(per_batch_distances)

        indices = (-per_batch_distances_arr).argsort()

        per_batch_distances_sorted = per_batch_distances_arr[indices]

        for l in self.label_list:
            per_label_distances[l] = sorted(per_label_distances[l], reverse=True)

        return per_batch_distances_sorted, per_label_distances

    @staticmethod
    def _proportional_sampling(label_list, E, Y_predicted, Y_original, window_size, n_windows, flag_replacement,
                               proportions_dict):
        per_label_E = {}
        per_label_Y_predicted = {}
        per_label_Y_original = {}

        # Dictionary to keep track of samples per label based on proportions
        n_samples_per_label = {str(l): int(proportions_dict[str(l)] * window_size) for l in label_list}
        total_samples = sum(n_samples_per_label.values())
        n_residual_samples = window_size - total_samples  # Adjust for any rounding errors in proportions

        for l in label_list:
            per_label_E[str(l)] = E[Y_original == l].copy()
            per_label_Y_predicted[str(l)] = Y_predicted[Y_original == l].copy()
            per_label_Y_original[str(l)] = Y_original[Y_original == l].copy()

        E_windows = []
        Y_predicted_windows = []
        Y_original_windows = []

        for i in range(n_windows):
            E_window_list = []
            Y_predicted_window_list = []
            Y_original_window_list = []

            for l in label_list:
                m_l = len(per_label_E[str(l)])
                n_samples = n_samples_per_label[str(l)]
                try:
                    l_idxs = np.random.choice(m_l, n_samples, replace=flag_replacement)
                except:
                    print(f"error: {l} , {m_l}")

                E_l_window = per_label_E[str(l)][l_idxs]
                Y_predicted_l_window = per_label_Y_predicted[str(l)][l_idxs]
                Y_original_l_window = per_label_Y_original[str(l)][l_idxs]

                E_window_list += E_l_window.tolist()
                Y_predicted_window_list += Y_predicted_l_window.tolist()
                Y_original_window_list += Y_original_l_window.tolist()

                if not flag_replacement:
                    # If not flag_replacement then remove vectors
                    per_label_E[str(l)] = np.delete(per_label_E[str(l)], l_idxs, 0)
                    per_label_Y_predicted[str(l)] = np.delete(per_label_Y_predicted[str(l)], l_idxs, 0)
                    per_label_Y_original[str(l)] = np.delete(per_label_Y_original[str(l)], l_idxs, 0)

            # Handling residual samples due to rounding errors in proportions
            if n_residual_samples > 0:
                labels, counts = np.unique(Y_original, return_counts=True)
                label_distribution = counts / counts.sum()
                additional_labels = np.random.choice(labels, n_residual_samples, replace=True, p=label_distribution)

                for l in additional_labels:
                    m_l = len(per_label_E[str(l)])
                    if m_l > 0:
                        idx = np.random.choice(m_l, 1, replace=False)
                        E_l_window = per_label_E[str(l)][idx]
                        Y_predicted_l_window = per_label_Y_predicted[str(l)][idx]
                        Y_original_l_window = per_label_Y_original[str(l)][idx]

                        E_window_list += E_l_window.tolist()
                        Y_predicted_window_list += Y_predicted_l_window.tolist()
                        Y_original_window_list += Y_original_l_window.tolist()

                        if not flag_replacement:
                            # If not flag_replacement then remove vectors
                            per_label_E[str(l)] = np.delete(per_label_E[str(l)], idx, 0)
                            per_label_Y_predicted[str(l)] = np.delete(per_label_Y_predicted[str(l)], idx, 0)
                            per_label_Y_original[str(l)] = np.delete(per_label_Y_original[str(l)], idx, 0)

            E_windows.append(np.array(E_window_list))
            Y_predicted_windows.append(np.array(Y_predicted_window_list))
            Y_original_windows.append(np.array(Y_original_window_list))

        return E_windows, Y_predicted_windows, Y_original_windows



    #TODO: improve this implementation
    @staticmethod
    def _balanced_sampling(label_list, E, Y_predicted, Y_original, window_size, n_windows, flag_replacement):

        per_label_E = {}
        per_label_Y_predicted = {}
        per_label_Y_original = {}

        for l in label_list:
            per_label_E[str(l)] = E[Y_original == l].copy()
            per_label_Y_predicted[str(l)] = Y_predicted[Y_original == l].copy()
            per_label_Y_original[str(l)] = Y_original[Y_original == l].copy()

        n_samples_per_label = window_size // len(label_list)
        n_residual_samples = window_size % len(label_list)

        E_windows = []
        Y_predicted_windows = []
        Y_original_windows = []
        for i in range(n_windows):
            E_window_list = []
            Y_predicted_window_list = []
            Y_original_window_list = []
            for l in label_list:
                m_l = len(per_label_E[str(l)])
                try:
                    l_idxs = np.random.choice(m_l, n_samples_per_label, replace=False)
                except:
                    print(f"error: {l} , {m_l}")
                E_l_window = per_label_E[str(l)][l_idxs]
                Y_predicted_l_window = per_label_Y_predicted[str(l)][l_idxs]
                Y_original_l_window = per_label_Y_original[str(l)][l_idxs]

                E_window_list += E_l_window.tolist()
                Y_predicted_window_list += Y_predicted_l_window.tolist()
                Y_original_window_list += Y_original_l_window.tolist()

                if bool(flag_replacement) == False:
                    # If not flag_replacement than remove vectors
                    per_label_E[str(l)] = np.delete(per_label_E[str(l)], l_idxs, 0)
                    per_label_Y_predicted[str(l)] = np.delete(per_label_Y_predicted[str(l)], l_idxs, 0)
                    per_label_Y_original[str(l)] = np.delete(per_label_Y_original[str(l)], l_idxs, 0)

            if n_residual_samples != 0:
                count_residual = 0
                while count_residual < n_residual_samples:

                    random_idx_l = np.random.choice(len(label_list), 1, replace=True)[0]
                    random_l = label_list[random_idx_l]

                    m_l = len(per_label_E[str(random_l)])
                    idx = np.random.choice(m_l, 1, replace=False)
                    E_l_window = per_label_E[str(random_l)][idx]
                    Y_predicted_l_window = per_label_Y_predicted[str(random_l)][idx]
                    Y_original_l_window = per_label_Y_original[str(random_l)][idx]

                    E_window_list += E_l_window.tolist()
                    Y_predicted_window_list += Y_predicted_l_window.tolist()
                    Y_original_window_list += Y_original_l_window.tolist()

                    count_residual += 1

                    if bool(flag_replacement) == False:
                        # If not flag_replacement than remove vectors
                        per_label_E[str(random_l)] = np.delete(per_label_E[str(random_l)], idx, 0)
                        per_label_Y_predicted[str(random_l)] = np.delete(per_label_Y_predicted[str(random_l)],
                                                                             idx, 0)
                        per_label_Y_original[str(random_l)] = np.delete(per_label_Y_original[str(random_l)],
                                                                            idx, 0)

            E_windows.append(np.array(E_window_list))
            Y_predicted_windows.append(np.array(Y_predicted_window_list))
            Y_original_windows.append(np.array(Y_original_window_list))


        return E_windows, Y_predicted_windows, Y_original_windows

    @staticmethod
    def _shuffle_dataset(E, Y_predicted, Y_original):
        p = np.random.permutation(len(E))
        E = E[p]
        Y_original = Y_original[p]
        Y_predicted = Y_predicted[p]
        return E, Y_predicted, Y_original



