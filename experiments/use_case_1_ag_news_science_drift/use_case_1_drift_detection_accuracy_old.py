import argparse
from alibi_detect.cd import KSDrift, MMDDrift, LSDDDrift, CVMDrift, ChiSquareDrift
from experiments.windows_manager.windows_generator import WindowsGenerator
from driftlens.driftlens import DriftLens
import os
import h5py
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import json
import datetime
import time
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--number_of_runs', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='bert')
    parser.add_argument('--window_size', type=int, default=2000)
    parser.add_argument('--number_of_windows', type=int, default=10)
    parser.add_argument('--drift_percentage', type=int, default=20)
    parser.add_argument('--batch_n_pc', type=int, default=150)
    parser.add_argument('--per_label_n_pc', type=int, default=75)
    parser.add_argument('--threshold_sensitivity', type=int, default=99)
    parser.add_argument('--threshold_number_of_estimation_samples', type=int, default=1000)
    parser.add_argument('--n_subsamples_sota', type=int, default=10000)
    parser.add_argument('--train_embedding_filepath', type=str, default=f"{os.getcwd()}/static/saved_embeddings/bert/train_embedding_0_1_2.hdf5")
    parser.add_argument('--test_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/bert/test_embedding_0_1_2.hdf5')
    parser.add_argument('--new_unseen_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/bert/new_unseen_embedding_0_1_2.hdf5')
    parser.add_argument('--drift_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/bert/drift_embedding_3.hdf5')
    parser.add_argument('--output_dir', type=str, default=f"{os.getcwd()}/static/outputs/bert/")
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def load_embedding(filepath, E_name=None, Y_original_name=None, Y_predicted_name=None):
    if filepath is not None:
        with h5py.File(filepath, "r") as hf:
            if E_name is None:
                E = hf["E"][()]
            else:
                E = hf[E_name][()]
            if Y_original_name is None:
                Y_original = hf["Y_original"][()]
            else:
                Y_original = hf[Y_original_name][()]
            if Y_predicted_name is None:
                Y_predicted = hf["Y_predicted"][()]
            else:
                Y_predicted = hf[Y_predicted_name][()]
    else:
        raise Exception("Error in loading the embedding file. Please set the embedding paths in the configuration file.")
    return E, Y_original, Y_predicted


def stratified_subsampling(E, Y, n_samples, unique_labels):
    # Calculate samples per class
    samples_per_class = int(n_samples / len(unique_labels))

    # Placeholder for stratified sample indices
    selected_indices = []

    for label in unique_labels:
        # Find indices where current label occurs
        label_indices = np.where(Y == label)[0]

        # If the class has fewer samples than samples_per_class, take them all
        # Otherwise, randomly choose samples_per_class from them
        if len(label_indices) <= samples_per_class:
            selected_indices.extend(label_indices)
        else:
            selected_indices.extend(np.random.choice(label_indices, samples_per_class, replace=False))

    # Now, selected_indices contains the indices of the stratified sample
    # Extract the corresponding elements from E
    E_subsample = E[selected_indices]

    return E_subsample, Y[selected_indices]


def main():
    print("Drift Detection Experiment - Use Case 1")

    # Parse arguments
    args = parse_args()

    print("Number of runs: ", args.number_of_runs)
    print("Window size: ", args.window_size)
    print("Number of windows: ", args.number_of_windows)
    print("Number of samples sota: ", args.n_subsamples_sota)
    print("Number of samples threshold: ", args.threshold_number_of_estimation_samples)
    print("Drift percentage: ", args.drift_percentage)

    training_label_list = [0, 1, 2]  # Labels used for training - 0: World, 1: Sports, 2: Business
    drift_label_list = [3]  # Labels used for drift simulation - 3: Science/technology

    if args.save_results:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        ts = time.time()
        timestamp = str(datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S'))
        output_filename = f"drift_detection_accuracy_model_{args.model_name}_win_size_{args.window_size}_n_windows_{args.number_of_windows}_drift_percentage_{args.drift_percentage}_{timestamp}.json"

    # Parse parameters
    window_size = args.window_size
    batch_n_pc = args.batch_n_pc
    per_label_n_pc = args.per_label_n_pc
    n_windows = args.number_of_windows

    # Load the embeddings
    E_train, Y_original_train, Y_predicted_train = load_embedding(args.train_embedding_filepath)
    E_test, Y_original_test, Y_predicted_test = load_embedding(args.test_embedding_filepath)
    E_new_unseen, Y_original_new_unseen, Y_predicted_new_unseen = load_embedding(args.new_unseen_embedding_filepath)
    E_drift, Y_original_drift, Y_predicted_drift = load_embedding(args.drift_embedding_filepath)

    ks_acc_list = []
    mmd_acc_list = []
    lsdd_acc_list = []
    cvm_acc_list = []
    chisquare_acc_list = []
    driftlens_acc_list = []

    output_dict_run_list = []

    for run_id in range(args.number_of_runs):
        print(f"Run {run_id + 1}/{args.number_of_runs}")

        # Initialize empty lists of predictions
        ks_preds = []
        mmd_preds = []
        lsdd_preds = []
        cvm_preds = []
        chisquare_preds = []
        dl_distances = []

        # Initialize the WindowsGenerator - used for creating the windows
        wg = WindowsGenerator(training_label_list,
                              drift_label_list,
                              E_new_unseen,
                              Y_predicted_new_unseen,
                              Y_original_new_unseen,
                              E_drift,
                              Y_predicted_drift,
                              Y_original_drift)

        # Initialize the DriftLens
        dl = DriftLens()

        # Estimate the baseline with DriftLens
        baseline = dl.estimate_baseline(E=E_train,
                                        Y=Y_predicted_train,
                                        label_list=training_label_list,
                                        batch_n_pc=batch_n_pc,
                                        per_label_n_pc=per_label_n_pc)

        # Estimate the threshold values with DriftLens
        per_batch_distances_sorted, per_label_distances_sorted = dl.random_sampling_threshold_estimation(label_list=training_label_list,
                                                                                                         E=E_test,
                                                                                                         Y=Y_predicted_test,
                                                                                                         batch_n_pc=batch_n_pc,
                                                                                                         per_label_n_pc=per_label_n_pc,
                                                                                                         window_size=window_size,
                                                                                                         n_samples=args.threshold_number_of_estimation_samples,
                                                                                                         flag_shuffle=True,
                                                                                                         flag_replacement=True)

        # Calculate the threshold values
        l = np.array(per_batch_distances_sorted)
        l = l[(l > np.quantile(l, 0.01)) & (l < np.quantile(l, 0.99))].tolist()
        per_batch_th = max(l)

        E_subsample, Y_subsample = stratified_subsampling(E_train,
                                                          Y_original_train,
                                                          n_samples=args.n_subsamples_sota,
                                                          unique_labels=training_label_list)

        print(len(E_subsample))

        # Initialize drift detectors used for comparison
        ks_detector = KSDrift(E_subsample, p_val=.05)
        mmd_detector = MMDDrift(E_subsample, p_val=.05, n_permutations=100, backend="pytorch")
        lsdd_detector = LSDDDrift(E_subsample, backend='pytorch', p_val=.05)
        cvm_detector = CVMDrift(E_subsample, p_val=.05)
        #chisquare_detector = ChiSquareDrift(E_subsample, p_val=0.05)

        # Ground truth
        if args.drift_percentage > 0:
            ground_truth = [1] * n_windows
        else:
            ground_truth = [0] * n_windows

        # Generate windows and predict drift
        for i in tqdm(range(n_windows)):

            if args.drift_percentage > 0:
                # Drift
                E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_constant_drift_windows_generation(window_size=window_size,
                                                                                                                    n_windows=1,
                                                                                                                    drift_percentage=float(args.drift_percentage/100),
                                                                                                                    flag_shuffle=True,
                                                                                                                    flag_replacement=True)

            else:
                # No Drift
                E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_without_drift_windows_generation(window_size=window_size,
                                                                                                                  n_windows=1,
                                                                                                                  flag_shuffle=True,
                                                                                                                  flag_replacement=True)

            # Compute the window distribution distances (Frechet Inception Distance) with DriftLens
            dl_distance = dl.compute_window_distribution_distances(E_windows[0], Y_predicted_windows[0])

            # Predict drift with the drift detectors used for comparison
            ks_pred = ks_detector.predict(E_windows[0])
            mmd_pred = mmd_detector.predict(E_windows[0])
            lsdd_pred = lsdd_detector.predict(E_windows[0], return_p_val=True, return_distance=True)
            cvm_pred = cvm_detector.predict(E_windows[0], drift_type='batch', return_p_val=True, return_distance=True)
            #chisquare_pred = chisquare_detector.predict(E_windows[0], drift_type='batch', return_p_val=True, return_distance=True)
            chisquare_pred = 0

            # Append the predictions to the lists
            ks_preds.append(ks_pred["data"]["is_drift"])
            mmd_preds.append(mmd_pred["data"]["is_drift"])
            lsdd_preds.append(lsdd_pred["data"]["is_drift"])
            cvm_preds.append(cvm_pred["data"]["is_drift"])
            #chisquare_preds.append(chisquare_pred["data"]["is_drift"])

            dl_distances.append(dl_distance)


        dl_preds = []
        for dl_distance in dl_distances:
            if dl_distance["per-batch"] > per_batch_th:
                dl_preds.append(1)
            else:
                dl_preds.append(0)

        # Calculate the accuracy of the drift detectors
        ks_acc = accuracy_score(ground_truth, ks_preds, normalize=True)
        mmd_acc = accuracy_score(ground_truth, mmd_preds, normalize=True)
        lsdd_acc = accuracy_score(ground_truth, lsdd_preds, normalize=True)
        cvm_acc = accuracy_score(ground_truth, cvm_preds, normalize=True)
        #chisquare_acc = accuracy_score(ground_truth, chisquare_preds, normalize=True)
        chisquare_acc = 0
        driftlens_acc = accuracy_score(ground_truth, dl_preds, normalize=True)

        ks_acc_list.append(ks_acc)
        mmd_acc_list.append(mmd_acc)
        lsdd_acc_list.append(lsdd_acc)
        cvm_acc_list.append(cvm_acc)
        chisquare_acc_list.append(chisquare_acc)
        driftlens_acc_list.append(driftlens_acc)

        print("MMD: ", mmd_acc)
        print("KS: ", ks_acc)
        print("LSDD: ", lsdd_acc)
        print("CVM: ", cvm_acc)
        #print("ChiSquare: ", chisquare_acc)
        print("DriftLens: ", driftlens_acc)

        # Create the output dictionary
        output_dict_run = {f"run_id":run_id, "KS": ks_acc, "MMD": mmd_acc, "LSDD": lsdd_acc, "CVM": cvm_acc, "ChiSquare": chisquare_acc, "DriftLens": driftlens_acc}
        output_dict_run_list.append(output_dict_run)

    # Create final output dictionary
    output_dict = {"params": vars(args),
                   "runs":output_dict_run_list,
                   "accuracy_list": {"KS": ks_acc_list, "MMD": mmd_acc_list, "LSDD": lsdd_acc_list, "CVM": cvm_acc_list, "ChiSquare": chisquare_acc_list, "DriftLens": driftlens_acc_list},
                      "mean_accuracy": {"KS": np.mean(ks_acc_list), "MMD": np.mean(mmd_acc_list), "LSDD": np.mean(lsdd_acc_list), "CVM": np.mean(cvm_acc_list), "ChiSquare": np.mean(chisquare_acc_list), "DriftLens": np.mean(driftlens_acc_list)},
                     "standard_deviation_accuracy": {"KS": np.std(ks_acc_list), "MMD": np.std(mmd_acc_list), "LSDD": np.std(lsdd_acc_list), "CVM": np.std(cvm_acc_list), "ChiSquare": np.std(chisquare_acc_list), "DriftLens": np.std(driftlens_acc_list)}}

    # Save the output dictionary
    with open(os.path.join(args.output_dir, output_filename), 'w') as fp:
        json.dump(output_dict, fp)

    return


if __name__ == "__main__":
    main()