from abc import ABC, abstractmethod
from sklearn.decomposition import PCA

import numpy as np
import _frechet_inception_distance as fid
import os
import pickle
import json


class BaselineClass:
    """ Baseline CLass: it contains all the attributes and methods of the baseline. """
    def __init__(self):
        self.label_list = None  # List of labels used to train the model
        self.batch_n_pc = None  # Number of principal components to reduce the embedding for the entire batch drift
        self.per_label_n_pc = None  # number of principal components to reduce embedding for the per-label drift

        self. mean_vectors_dict = {}  # Dict containing the mean vectors: 1) per-label ["per-label"] and 2) for the entire batch ["batch"]
        self. covariance_matrices_dict = {}  # Dict containing the covariance matrices: 1) per-label ["per-label"] and 2) for the entire batch ["batch"]
        self.PCA_models_dict = {}  # Dict containing the PCA models: 1) per-label ["per-label"] and 2) for the entire batch ["batch"]
        self.n_samples_dict = {}  # Dict containing the number of samples: 1) per-label ["per-label"] and 2) in the entire batch ["batch"]

        self.mean_vectors_dict["per-label"] = {}  # Dict containing the per-label mean vectors ["label"]: mean(label)
        self.covariance_matrices_dict["per-label"] = {}  # Dict containing the per-label covariance matrices ["label"]: covariance(label)
        self.PCA_models_dict["per-label"] = {}  # Dict containing the per-label PCA models ["label"]: PCA(label)
        self.n_samples_dict["per-label"] = {}  # Dict containing the per-label number of samples

        self.mean_vectors_dict["batch"] = None
        self.covariance_matrices_dict["batch"] = None
        self.PCA_models_dict["batch"] = None
        self.n_samples_dict["batch"] = None

        self.description = ""
        return

    def fit(self, label_list, batch_n_pc, per_label_n_pc,
            per_label_mean_dict, per_label_covariance_dict, per_label_PCA_models, per_label_n_samples,
            batch_mean_vector=None, batch_covariance_matrix=None, batch_PCA_model=None, batch_n_samples=None, description=""):
        """ Fits the baseline attributes.
        Args:
            label_list (list): List of labels used to train the model.
            batch_n_pc (int): Number of principal components to reduce the embedding for the entire batch drift.
            per_label_n_pc (int): Number of principal components to reduce embedding for the per-label drift.
            per_label_mean_dict (dict): Dict containing the per-label mean vectors ["label"]: mean(label).
            per_label_covariance_dict (dict): Dict containing the per-label covariance matrices ["label"]: covariance(label).
            per_label_PCA_models (dict): Dict containing the per-label PCA models ["label"]: PCA(label).
            per_label_n_samples (dict): Dict containing the per-label number of samples.
            batch_mean_vector (np.array): Mean vector for the entire batch.
            batch_covariance_matrix (np.array): Covariance matrix for the entire batch.
            batch_PCA_model (PCA): PCA model for the entire batch.
            batch_n_samples (int): Number of samples in the entire batch.
            description (str): Description of the baseline.
        Returns:
            None
        """
        self.label_list = label_list
        self.batch_n_pc = batch_n_pc
        self.per_label_n_pc = per_label_n_pc
        self.mean_vectors_dict["batch"] = batch_mean_vector
        self.covariance_matrices_dict["batch"] = batch_covariance_matrix
        self.PCA_models_dict["batch"] = batch_PCA_model
        self.n_samples_dict["batch"] = batch_n_samples
        self.description = description

        for label in self.label_list:
            self.mean_vectors_dict["per-label"][str(label)] = per_label_mean_dict[str(label)]
            self.covariance_matrices_dict["per-label"][str(label)] = per_label_covariance_dict[str(label)]
            self.PCA_models_dict["per-label"][str(label)] = per_label_PCA_models[str(label)]
            self.n_samples_dict["per-label"][str(label)] = per_label_n_samples[str(label)]

        return

    def save(self, folder_path, baseline_name):
        """ Saves persistently on disk the baseline.
        Args:
            folder_path (str): Folder path where save the baseline.
            baseline_name (str): Filename of the baseline folder.
        Returns:
            (str): Baseline folder path.
        """
        BASELINE_PATH = os.path.join(folder_path, baseline_name)
        BASELINE_PCA_FOLDER = os.path.join(BASELINE_PATH, "pca_models")
        BASELINE_STATISTICS_FOLDER = os.path.join(BASELINE_PATH, "saved_statistics")

        # Create the baseline folder
        if not os.path.exists(BASELINE_PATH):
            os.makedirs(BASELINE_PATH)

        # Create the PCA folder
        if not os.path.exists(BASELINE_PCA_FOLDER):
            os.makedirs(BASELINE_PCA_FOLDER)

        # Save each per-label trained PCA
        for pca_key, pca_value in self.PCA_models_dict["per-label"].items():
            filename = "baseline_pca_l_{}.pkl".format(pca_key)
            with open(os.path.join(BASELINE_PCA_FOLDER, filename), 'wb') as pickle_file:
                pickle.dump(pca_value, pickle_file)

        # Save the entire batch trained PCA
        with open(os.path.join(BASELINE_PCA_FOLDER, "baseline_pca_batch.pkl"), 'wb') as pickle_file:
            pickle.dump(self.PCA_models_dict["batch"], pickle_file)

        # Create the statistics folder
        if not os.path.exists(BASELINE_STATISTICS_FOLDER):
            os.makedirs(BASELINE_STATISTICS_FOLDER)

        # Save each per-label mean vector
        for key, value in self.mean_vectors_dict["per-label"].items():
            outfile = os.path.join(BASELINE_STATISTICS_FOLDER, "baseline_mean_l_{}".format(str(key)))
            np.save(outfile, value)

        # Save the entire batch mean vector
        outfile = os.path.join(BASELINE_STATISTICS_FOLDER, "baseline_mean_batch")
        np.save(outfile, self.mean_vectors_dict["batch"])

        # Save each per-label covariance matrix
        for key, value in self.covariance_matrices_dict["per-label"].items():
            outfile = os.path.join(BASELINE_STATISTICS_FOLDER, "baseline_covariance_l_{}".format(str(key)))
            np.save(outfile, value)

        # Save the entire batch covariance matrix
        outfile = os.path.join(BASELINE_STATISTICS_FOLDER, "baseline_covariance_batch")
        np.save(outfile, self.covariance_matrices_dict["batch"])

        baseline_info_dict = {"label_list": self.label_list, "per_label_n_pc": self.per_label_n_pc, "batch_n_pc": self.batch_n_pc,
                              "n_samples_dict": self.n_samples_dict, "description": self.description}
        try:
            with open(os.path.join(BASELINE_PATH, 'baseline_info.json'), 'w') as fp:
                json.dump(baseline_info_dict, fp)
        except Exception as e:
            raise Exception(f'Error in saving the baseline info json: {e}')
        return BASELINE_PATH

    def load(self, folder_path, baseline_name):
        """ Loads the baseline from folder.
        Args:
            folder_path (str): Folder path where the baseline is saved.
            baseline_name (str): Filename of the baseline folder.
        Returns:
            None
        """
        BASELINE_PATH = os.path.join(folder_path, baseline_name)
        BASELINE_PCA_FOLDER = os.path.join(BASELINE_PATH, "pca_models")
        BASELINE_STATISTICS_FOLDER = os.path.join(BASELINE_PATH, "saved_statistics")

        try:
            with open(os.path.join(BASELINE_PATH, 'baseline_info.json')) as json_file:
                info_dict = json.load(json_file)

            self.label_list = info_dict["label_list"]
            self.batch_n_pc = info_dict["batch_n_pc"]
            self.per_label_n_pc = info_dict["per_label_n_pc"]
            self.description = info_dict["description"]
            self.n_samples_dict = info_dict["n_samples_dict"]
        except:
            raise Exception(f'Error in loading the baseline. "baseline_info.json" not found.')

        for label in self.label_list:
            pca_l_filename = "baseline_pca_l_{}.pkl".format(str(label))
            self.PCA_models_dict["per-label"][str(label)] = pickle.load(open(os.path.join(BASELINE_PCA_FOLDER, pca_l_filename), 'rb'))

            mean_l_filename = "baseline_mean_l_{}.npy".format(str(label))
            self.mean_vectors_dict["per-label"][str(label)] = np.load(os.path.join(BASELINE_STATISTICS_FOLDER, mean_l_filename))

            covariance_l_filename = "baseline_covariance_l_{}.npy".format(str(label))
            self.covariance_matrices_dict["per-label"][str(label)] = np.load(os.path.join(BASELINE_STATISTICS_FOLDER, covariance_l_filename))

        self.PCA_models_dict["batch"] = pickle.load(open(os.path.join(BASELINE_PCA_FOLDER, "baseline_pca_batch.pkl"), 'rb'))
        self.mean_vectors_dict["batch"] = np.load(os.path.join(BASELINE_STATISTICS_FOLDER, "baseline_mean_batch.npy"))
        self.covariance_matrices_dict["batch"] = np.load(os.path.join(BASELINE_STATISTICS_FOLDER, "baseline_covariance_batch.npy"))

        return

    def get_PCA_model_by_label(self, label):
        """ Gets the PCA model of a label (per-label).
        Args:
            label (int): Label.
        Returns:
            PCA: PCA model of the label.
        """
        return self.PCA_models_dict["per-label"][str(label)]

    def get_mean_vector_by_label(self, label):
        """ Gets the mean vector of a label (per-label).
        Args:
            label (int): Label.
        Returns:
            np.ndarray: Mean vector of the label.
        """
        return self.mean_vectors_dict["per-label"][str(label)]

    def get_covariance_matrix_by_label(self, label):
        """ Gets the covariance matrix of a label (per-label).
        Args:
            label (int): Label.
        Returns:
            np.ndarray: Covariance matrix of the label.
        """
        return self.covariance_matrices_dict["per-label"][str(label)]

    def get_n_samples_by_label(self, label):
        """ Gets the number of samples of a label (per-label).
        Args:
            label (int): Label.
        Returns:
            int: Number of samples of the label.
        """
        return self.n_samples_dict["per-label"][str(label)]

    def get_per_label_mean_vectors(self):
        """ Gets the mean vectors of each label (per-label).
        Returns:
            dict: Mean vectors of each label.
        """
        return self.mean_vectors_dict["per-label"]

    def get_per_label_covariance_matrices(self):
        """ Gets the covariance matrices of each label (per-label).
        Returns:
            dict: Covariance matrices of each label.
        """
        return self.covariance_matrices_dict["per-label"]

    def get_per_label_n_samples(self):
        """ Gets the number of samples per label.
        Returns:
            dict: Number of samples per label.
        """
        return self.n_samples_dict["per-label"]

    def get_batch_PCA_model(self):
        """ Gets the PCA model of the entire batch.
        Returns:
            PCA: PCA model of the entire batch (per-batch).
        """
        return self.PCA_models_dict["batch"]

    def get_batch_mean_vector(self):
        """ Gets the mean vector of the entire batch.
        Returns:
            np.ndarray: Mean vector of the entire batch (per-batch).
        """
        return self.mean_vectors_dict["batch"]

    def get_batch_covariance_matrix(self):
        """ Gets the covariance matrix of the entire batch.
        Returns:
            np.ndarray: Covariance matrix of the entire batch (per-batch).
        """
        return self.covariance_matrices_dict["batch"]

    def get_batch_n_samples(self):
        """ Gets the number of samples of the entire batch (per-batch).
        Returns:
            int: Number of samples of the entire batch.
        """
        return self.n_samples_dict["batch"]

    def get_per_label_number_of_principal_components(self):
        """ Gets the number of principal components for PCA for each label (per-label).
        Returns:
            dict: Number of principal components for PCA for each label.
        """
        return self.per_label_n_pc

    def get_batch_number_of_principal_components(self):
        """ Gets the number of principal components for PCA for the entire batch (per-batch).
        Returns:
            int: Number of principal components for PCA for the entire batch.
        """
        return self.batch_n_pc

    def get_label_list(self):
        """ Gets the list of labels used to train the model.
        Returns:
            list: List of labels used to train the model.
        """
        return self.label_list

    def get_description(self):
        """ Gets the description of the baseline estimator.
        Returns:
            str: Description of the baseline estimator.
        """
        return self.description

    def set_description(self, description):
        """ Sets the description of the baseline estimator.
        Args:
            description (str): Description of the baseline estimator.
        """
        self.description = description
        return


class BaselineEstimatorMethod(ABC):
    """ Abstract Baseline Estimator Method class. """

    def __init__(self, label_list, batch_n_pc, per_label_n_pc):
        self.batch_n_pc = batch_n_pc  # Number of principal components for PCA for the entire batch
        self.per_label_n_pc = per_label_n_pc  # Number of principal components for PCA for the per-label
        self.label_list = label_list  # List of labels used to train the model
        return

    @abstractmethod
    def estimate_baseline(self, **kwargs) -> BaselineClass:
        """ Abstract Method: Estimates the baseline. """
        pass

    def _fit_pca(self, E, Y):
        """ Fits a PCA for each label and for the entire batch.

        Args:
             E (np.array): Embedding vectors of shape (m, n_e), where m is the number of samples and n_e the embedding dimensionality.
             Y (np.array): Labels (predicted/original) of shape (m, 1), where m is the number of samples.
         Returns:
             sklearn.decomposition.PCA: the PCA computed over the entire batch.
             dict: dictionary containing the per-label PCA fitted for each label {'label': PCA_l}.
        """
        # Fit the PCA for the entire batch of data (per-batch)
        batch_PCA = PCA(n_components=self.batch_n_pc)
        batch_PCA.fit(E)

        # Fit a per-label PCA for each label - "l": PCA_l
        per_label_PCA_dict = {}

        for label in self.label_list:

            # Select examples of the current label (predicted/original)
            E_l_idxs = np.nonzero(Y == label)  # Indices of embedding vectors E for label l
            E_l = E[E_l_idxs]  # Embedding vectors E for label l

            # Fit PCA with baseline examples of the current label
            per_label_PCA = PCA(n_components=self.per_label_n_pc)
            per_label_PCA.fit(E_l)

            # Store the PCA model in the dictionary
            per_label_PCA_dict[str(label)] = per_label_PCA

        return batch_PCA, per_label_PCA_dict


class StandardBaselineEstimator(BaselineEstimatorMethod):
    """ Standard Baseline Estimator Class: Implementation of the BaselineEstimatorMethod Abstract Class. """

    def __init__(self, label_list, batch_n_pc, per_label_n_pc):
        BaselineEstimatorMethod.__init__(self, label_list, batch_n_pc, per_label_n_pc)
        return

    def estimate_baseline(self, E, Y):
        """ Estimates the baseline.

         Args:
             E (np.array): Embedding vectors of shape (m, n_e), where m is the number of samples and n_e the embedding dimensionality.
             Y (np.array): Labels (predicted/origianl) of shape (m, 1), where m is the number of samples.
         Returns:
             BaselineClass: Returns the baseline objects with the estimated models.
         """
        # Fit PCAs from the embedding vectors
        batch_PCA, per_label_PCA_dict = self._fit_pca(E, Y)

        # Reduce the embedding dimensionality of the entire batch with the batch_PCA
        E_reduced = batch_PCA.transform(E)

        # Estimate the mean vector and the covariance matrix for the entire batch
        batch_mean = fid.get_mean(E_reduced)
        batch_covariance = fid.get_covariance(E_reduced)

        # Counts the batch number of samples
        batch_n_samples = len(Y)

        # Dictionary containing the per-label mean vector - "l": mean_l
        per_label_mean_dict = {}

        # Dictionary containing the per-label covariance matrix - "l": covariance_l
        per_label_covariance_dict = {}

        # Dictionary containing the per-label number of samples - "l": covariance_l
        per_label_n_samples_dict = {}

        for label in self.label_list:
            # Select examples of the current label (predicted/original)
            E_l_idxs = np.nonzero(Y == label)
            E_l = E[E_l_idxs]

            # Reduce the embedding dimensionality with PCA_l
            E_l_reduced = per_label_PCA_dict[str(label)].transform(E_l)

            # Estimate the mean vector and the covariance matrix for the label l
            mean_l = fid.get_mean(E_l_reduced)
            covariance_l = fid.get_covariance(E_l_reduced)

            # Store the mean vector and the covariance matrix in the dictionary
            per_label_mean_dict[str(label)] = mean_l
            per_label_covariance_dict[str(label)] = covariance_l

            # Store the number of per-label samples in the dictionary
            per_label_n_samples_dict[str(label)] = len(E_l_reduced)

        # Create a Baseline object
        baseline = BaselineClass()

        # Fit the Baseline info
        baseline.fit(self.label_list, self.batch_n_pc, self.per_label_n_pc,
                     per_label_mean_dict, per_label_covariance_dict, per_label_PCA_dict, per_label_n_samples_dict,
                     batch_mean, batch_covariance, batch_PCA, batch_n_samples)

        return baseline