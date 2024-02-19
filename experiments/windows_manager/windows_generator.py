import numpy as np
from numpy import random


def create_proportions_dict(Y):
    labels, counts = np.unique(Y, return_counts=True)
    total_count = counts.sum()
    proportions = {str(label): count / total_count for label, count in zip(labels, counts)}
    return proportions

class WindowsGenerator:

    def __init__(self, training_label_list, drifted_label_list, E, Y_predicted, Y_original, E_drifted, Y_predicted_drifted, Y_original_drifted):
        self.training_label_list = training_label_list
        self.drifted_label_list = drifted_label_list
        self.E = E
        self.Y_predicted = Y_predicted
        self.Y_original = Y_original

        self.E_drifted = E_drifted
        self.Y_original_drifted = Y_original_drifted
        self.Y_predicted_drifted = Y_predicted_drifted
        return

    def custom_windows_generation(self, window_size, drift_pattern_dict, flag_shuffle=True, flag_replacement=False):
        if flag_shuffle:
            self.shuffle_datasets()

        per_label_E = {}
        per_label_Y_predicted = {}
        per_label_Y_original = {}

        for l in self.training_label_list:
            per_label_E[str(l)] = self.E[self.Y_original == l].copy()
            per_label_Y_predicted[str(l)] = self.Y_predicted[self.Y_original == l].copy()
            per_label_Y_original[str(l)] = self.Y_original[self.Y_original == l].copy()

        for l in self.drifted_label_list:
            per_label_E[str(l)] = self.E_drifted[self.Y_original_drifted == l].copy()
            per_label_Y_predicted[str(l)] = self.Y_predicted_drifted[self.Y_original_drifted == l].copy()
            per_label_Y_original[str(l)] = self.Y_original_drifted[self.Y_original_drifted == l].copy()

        E_windows = []
        Y_predicted_windows = []
        Y_original_windows = []
        for drift_pattern_window in drift_pattern_dict:
            E_window_list = []
            Y_predicted_window_list = []
            Y_original_window_list = []

            for l, percentage_l in drift_pattern_window.items():

                if (l in per_label_E) and (percentage_l > 0):
                    n_samples_l = int(window_size * percentage_l)

                    m_l = len(per_label_E[str(l)])
                    l_idxs = np.random.choice(m_l, n_samples_l, replace=False)
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

            E_windows.append(np.array(E_window_list))
            Y_predicted_windows.append(np.array(Y_predicted_window_list))
            Y_original_windows.append(np.array(Y_original_window_list))

        return E_windows, Y_predicted_windows, Y_original_windows


    def balanced_without_drift_windows_generation(self, window_size, n_windows, flag_shuffle=True, flag_replacement=False):

        if bool(flag_shuffle):
            self.shuffle_datasets()

        E_windows, Y_predicted_windows, Y_original_windows = self._balanced_sampling(self.training_label_list,
                                                                                     self.E,
                                                                                     self.Y_predicted,
                                                                                     self.Y_original,
                                                                                     window_size,
                                                                                     n_windows,
                                                                                     flag_replacement)
        for i in range(n_windows):
            E_windows[i], Y_predicted_windows[i], Y_original_windows[i] = self._shuffle_dataset(E_windows[i],
                                                                                                Y_predicted_windows[i],
                                                                                                Y_original_windows[i])
        return E_windows, Y_predicted_windows, Y_original_windows

    def proportional_without_drift_windows_generation(self, window_size, n_windows, proportions_dict, flag_shuffle=True, flag_replacement=False):

        if bool(flag_shuffle):
            self.shuffle_datasets()

        E_windows, Y_predicted_windows, Y_original_windows = self._proportional_sampling(self.training_label_list,
                                                                                         self.E,
                                                                                         self.Y_predicted,
                                                                                         self.Y_original,
                                                                                         window_size=window_size,
                                                                                         n_windows=n_windows,
                                                                                         flag_replacement=flag_replacement,
                                                                                         proportions_dict=proportions_dict)
        for i in range(n_windows):
            E_windows[i], Y_predicted_windows[i], Y_original_windows[i] = self._shuffle_dataset(E_windows[i],
                                                                                                Y_predicted_windows[i],
                                                                                                Y_original_windows[i])
        return E_windows, Y_predicted_windows, Y_original_windows

    def balanced_constant_drift_windows_generation(self, window_size, n_windows, drift_percentage, flag_shuffle=True, flag_replacement=False):
        if bool(flag_shuffle):
            self.shuffle_datasets()


        m_window_drifted = int(round(window_size*drift_percentage))
        m_window = int(window_size-m_window_drifted)

        E_windows, Y_predicted_windows, Y_original_windows = self._balanced_sampling(self.training_label_list,
                                                                                     self.E,
                                                                                     self.Y_predicted,
                                                                                     self.Y_original,
                                                                                     m_window,
                                                                                     n_windows,
                                                                                     flag_replacement)

        E_windows_drifted, Y_predicted_windows_drifted, Y_original_windows_drifted = self._balanced_sampling(self.drifted_label_list,
                                                                                     self.E_drifted,
                                                                                     self.Y_predicted_drifted,
                                                                                     self.Y_original_drifted,
                                                                                     m_window_drifted,
                                                                                     n_windows,
                                                                                     flag_replacement)
        for i in range(n_windows):
            E_windows[i] = np.concatenate((E_windows[i], E_windows_drifted[i]), axis=0)
            Y_predicted_windows[i] = np.concatenate((Y_predicted_windows[i], Y_predicted_windows_drifted[i]), axis=0)
            Y_original_windows[i] = np.concatenate((Y_original_windows[i], Y_original_windows_drifted[i]), axis=0)

            E_windows[i], Y_predicted_windows[i], Y_original_windows[i] = self._shuffle_dataset(E_windows[i], Y_predicted_windows[i], Y_original_windows[i])

        return E_windows, Y_predicted_windows, Y_original_windows

    def balanced_incremental_drift_windows_generation(self, window_size, n_windows, starting_drift_percentage, drift_increase_rate, drift_offset=0, flag_shuffle=True, flag_replacement=False):
        if bool(flag_shuffle):
            self.shuffle_datasets()

        drift_percentage = 0

        E_windows = []
        Y_predicted_windows = []
        Y_original_windows = []
        for i in range(n_windows):

            if i == drift_offset:
                drift_percentage = starting_drift_percentage

            m_window_drifted = int(round(window_size * drift_percentage))
            m_window = int(window_size - m_window_drifted)

            if m_window > 0:
                E_window, Y_predicted_window, Y_original_window = self._balanced_sampling(self.training_label_list,
                                                                                          self.E,
                                                                                          self.Y_predicted,
                                                                                          self.Y_original,
                                                                                          m_window,
                                                                                          1,
                                                                                          flag_replacement)


            if m_window_drifted > 0:
                E_window_drifted, Y_predicted_window_drifted, Y_original_window_drifted = self._balanced_sampling(self.drifted_label_list,
                                                                                                                  self.E_drifted,
                                                                                                                  self.Y_predicted_drifted,
                                                                                                                  self.Y_original_drifted,
                                                                                                                  m_window_drifted,
                                                                                                                  1,
                                                                                                                  flag_replacement)

                if m_window > 0:
                    E_windows.append(np.concatenate((E_window[0], E_window_drifted[0]), axis=0))
                    Y_predicted_windows.append(np.concatenate((Y_predicted_window[0], Y_predicted_window_drifted[0]), axis=0))
                    Y_original_windows.append(np.concatenate((Y_original_window[0], Y_original_window_drifted[0]), axis=0))
                else:
                    E_windows.append(E_window_drifted[0])
                    Y_predicted_windows.append(Y_predicted_window_drifted[0])
                    Y_original_windows.append(Y_original_window_drifted[0])
            else:
                E_windows.append(E_window[0])
                Y_predicted_windows.append(Y_predicted_window[0])
                Y_original_windows.append(Y_original_window[0])

            if i >= drift_offset:
                drift_percentage += drift_increase_rate
                drift_percentage = float(min(drift_percentage, 1.0))

        for i in range(n_windows):
            E_windows[i], Y_predicted_windows[i], Y_original_windows[i] = self._shuffle_dataset(E_windows[i], Y_predicted_windows[i], Y_original_windows[i])

        return E_windows, Y_predicted_windows, Y_original_windows

    def proportional_constant_drift_windows_generation(self, window_size, n_windows, drift_percentage, proportions_dict, flag_shuffle=True, flag_replacement=False):
        if bool(flag_shuffle):
            self.shuffle_datasets()


        m_window_drifted = int(round(window_size*drift_percentage))
        m_window = int(window_size-m_window_drifted)

        E_windows, Y_predicted_windows, Y_original_windows = self._proportional_sampling(self.training_label_list,
                                                                                     self.E,
                                                                                     self.Y_predicted,
                                                                                     self.Y_original,
                                                                                     m_window,
                                                                                     n_windows,
                                                                                     flag_replacement,
                                                                                     proportions_dict)

        E_windows_drifted, Y_predicted_windows_drifted, Y_original_windows_drifted = self._balanced_sampling(self.drifted_label_list,
                                                                                     self.E_drifted,
                                                                                     self.Y_predicted_drifted,
                                                                                     self.Y_original_drifted,
                                                                                     m_window_drifted,
                                                                                     n_windows,
                                                                                     flag_replacement)
        for i in range(n_windows):
            E_windows[i] = np.concatenate((E_windows[i], E_windows_drifted[i]), axis=0)
            Y_predicted_windows[i] = np.concatenate((Y_predicted_windows[i], Y_predicted_windows_drifted[i]), axis=0)
            Y_original_windows[i] = np.concatenate((Y_original_windows[i], Y_original_windows_drifted[i]), axis=0)

            E_windows[i], Y_predicted_windows[i], Y_original_windows[i] = self._shuffle_dataset(E_windows[i], Y_predicted_windows[i], Y_original_windows[i])

        return E_windows, Y_predicted_windows, Y_original_windows

    def balanced_periodic_drift_windows_generation(self, window_size, n_windows, drift_offset, drift_duration, drift_percentage, flag_shuffle=True, flag_replacement=False):
        if bool(flag_shuffle):
            self.shuffle_datasets()

        n_periodic = n_windows//(drift_offset+drift_duration)
        n_periodic_remainder = n_windows%(drift_offset+drift_duration)

        E_window_periodic = []
        Y_predicted_periodic = []
        Y_original_periodic = []

        if n_periodic > 0:
            for i in range(n_periodic):

                E_window_no_drift, Y_predicted_window_no_drift, Y_original_window_no_drift = self._balanced_sampling(label_list=self.training_label_list,
                                                                                                                      E=self.E,
                                                                                                                      Y_predicted=self.Y_predicted,
                                                                                                                      Y_original=self.Y_original,
                                                                                                                      window_size=window_size,
                                                                                                                      n_windows=drift_offset,
                                                                                                                      flag_replacement=flag_replacement)

                E_window_periodic.extend(E_window_no_drift)
                Y_predicted_periodic.extend(Y_predicted_window_no_drift)
                Y_original_periodic.extend(Y_original_window_no_drift)

                m_window_drifted = int(round(window_size * drift_percentage))
                m_window = int(window_size - m_window_drifted)

                E_windows, Y_predicted_windows, Y_original_windows = self._balanced_sampling(self.training_label_list,
                                                                                             self.E,
                                                                                             self.Y_predicted,
                                                                                             self.Y_original,
                                                                                             m_window,
                                                                                             drift_duration,
                                                                                             flag_replacement)


                E_windows_drifted, Y_predicted_windows_drifted, Y_original_windows_drifted = self._balanced_sampling(
                                                                                                                self.drifted_label_list,
                                                                                                                self.E_drifted,
                                                                                                                self.Y_predicted_drifted,
                                                                                                                self.Y_original_drifted,
                                                                                                                m_window_drifted,
                                                                                                                drift_duration,
                                                                                                                flag_replacement)
                for i in range(drift_duration):
                    E_windows[i] = np.concatenate((E_windows[i], E_windows_drifted[i]), axis=0)
                    Y_predicted_windows[i] = np.concatenate((Y_predicted_windows[i], Y_predicted_windows_drifted[i]), axis=0)
                    Y_original_windows[i] = np.concatenate((Y_original_windows[i], Y_original_windows_drifted[i]), axis=0)

                    E_windows[i], Y_predicted_windows[i], Y_original_windows[i] = self._shuffle_dataset(E_windows[i], Y_predicted_windows[i], Y_original_windows[i])

                E_window_periodic.extend(E_windows)
                Y_predicted_periodic.extend(Y_predicted_windows)
                Y_original_periodic.extend(Y_original_windows)

            if n_periodic_remainder > 0:
                if n_periodic_remainder <= drift_offset:
                    drift_offset_remainder = n_periodic_remainder
                    drift_duration_remainder = 0
                else:
                    drift_offset_remainder = drift_offset
                    drift_duration_remainder = n_periodic_remainder - drift_offset_remainder

                E_window_no_drift, Y_predicted_window_no_drift, Y_original_window_no_drift = self._balanced_sampling(label_list=self.training_label_list,
                                                                                                                    E=self.E,
                                                                                                                    Y_predicted=self.Y_predicted,
                                                                                                                    Y_original=self.Y_original,
                                                                                                                    window_size=window_size,
                                                                                                                    n_windows=drift_offset_remainder,
                                                                                                                    flag_replacement=flag_replacement)

                E_window_periodic.extend(E_window_no_drift)
                Y_predicted_periodic.extend(Y_predicted_window_no_drift)
                Y_original_periodic.extend(Y_original_window_no_drift)

                if drift_duration_remainder > 0:
                    m_window_drifted = int(round(window_size * drift_percentage))
                    m_window = int(window_size - m_window_drifted)

                    E_windows, Y_predicted_windows, Y_original_windows = self._balanced_sampling(self.training_label_list,
                                                                                                self.E,
                                                                                                self.Y_predicted,
                                                                                                self.Y_original,
                                                                                                m_window,
                                                                                                drift_duration_remainder,
                                                                                                flag_replacement)

                    E_windows_drifted, Y_predicted_windows_drifted, Y_original_windows_drifted = self._balanced_sampling(self.drifted_label_list,
                                                                                                                        self.E_drifted,
                                                                                                                        self.Y_predicted_drifted,
                                                                                                                        self.Y_original_drifted,
                                                                                                                        m_window_drifted,
                                                                                                                        drift_duration_remainder,
                                                                                                                        flag_replacement)

                    for i in range(drift_duration_remainder):
                        E_windows[i] = np.concatenate((E_windows[i], E_windows_drifted[i]), axis=0)
                        Y_predicted_windows[i] = np.concatenate((Y_predicted_windows[i], Y_predicted_windows_drifted[i]), axis=0)
                        Y_original_windows[i] = np.concatenate((Y_original_windows[i], Y_original_windows_drifted[i]), axis=0)

                        E_windows[i], Y_predicted_windows[i], Y_original_windows[i] = self._shuffle_dataset(E_windows[i], Y_predicted_windows[i], Y_original_windows[i])

                    E_window_periodic.extend(E_windows)
                    Y_predicted_periodic.extend(Y_predicted_windows)
                    Y_original_periodic.extend(Y_original_windows)
        return E_window_periodic, Y_predicted_periodic, Y_original_periodic


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

            print(create_proportions_dict(Y_original_window_list))

        return E_windows, Y_predicted_windows, Y_original_windows


    @staticmethod
    def _shuffle_dataset(E, Y_predicted, Y_original):
        p = np.random.permutation(len(E))
        E = E[p]
        Y_original = Y_original[p]
        Y_predicted = Y_predicted[p]
        return E, Y_predicted, Y_original

    def shuffle_datasets(self):
        p = np.random.permutation(len(self.E))
        self.E = self.E[p]
        self.Y_original = self.Y_original[p]
        self.Y_predicted = self.Y_predicted[p]

        p_drifted = np.random.permutation(len(self.E_drifted))
        self.E_drifted = self.E_drifted[p_drifted]
        self.Y_original_drifted = self.Y_original_drifted[p_drifted]
        self.Y_predicted_drifted = self.Y_predicted_drifted[p_drifted]
        return




if __name__ == '__main__':
    dataset_name = "20_newsgroups"
    use_case_name = "bert_0_1_3"

    training_label_list = [0, 1, 2]
    drifted_label_list = [3, 4]

    embedding_size = 768

    baseline_dataset = "test"
    baseline_name = "baseline_test"

    train_size = 10000
    test_size = 5000
    new_unseen_size = 5000
    drifted_size = 3000

    batch_n_pc = 100
    per_label_n_pc = 50

    E_train = random.rand(train_size, embedding_size)
    Y_train = np.random.choice(3, train_size)

    E_test = random.rand(test_size, embedding_size)
    Y_test = np.random.choice(3, test_size)

    E_new_unseen = random.rand(new_unseen_size, embedding_size)
    Y_original_new_unseen = np.random.choice(3, new_unseen_size)
    Y_predicted_new_unseen = Y_original_new_unseen

    E_drifted = random.rand(drifted_size, embedding_size)
    Y_original_drifted = np.random.choice(2, drifted_size)
    Y_original_drifted = Y_original_drifted+3
    Y_predicted_drifted = Y_original_drifted


    wg = WindowsGenerator(training_label_list, drifted_label_list, E_new_unseen, Y_predicted_new_unseen, Y_original_new_unseen,
                          E_drifted, Y_predicted_drifted, Y_original_drifted)

    #E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_without_drift_windows_generation(40, 3, flag_shuffle=True, flag_replacement=False)
    #E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_constant_drift_windows_generation(40, 3, 0.10, flag_shuffle=True, flag_replacement=False)
    E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_incremental_drift_windows_generation(window_size=40,
                                                                                                          n_windows=10,
                                                                                                          starting_drift_percentage=0.1,
                                                                                                          drift_increase_rate=0.25,
                                                                                                          drift_offset=4,
                                                                                                          flag_shuffle=True,
                                                                                                          flag_replacement=True)

    print(E_windows)
    print("end main")