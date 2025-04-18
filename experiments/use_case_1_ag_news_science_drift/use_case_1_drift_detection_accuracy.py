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
import statistics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--number_of_runs', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='bert')
    parser.add_argument('--window_size', type=int, default=2000)
    parser.add_argument('--number_of_windows', type=int, default=10)
    parser.add_argument('--drift_percentage', type=int, nargs='+', default=[0, 5, 10, 15, 20]),
    parser.add_argument('--batch_n_pc', type=int, default=150)
    parser.add_argument('--per_label_n_pc', type=int, default=75)
    parser.add_argument('--threshold_sensitivity', type=int, default=1)
    parser.add_argument('--threshold_number_of_estimation_samples', type=int, default=10000)
    parser.add_argument('--n_subsamples_mmd', type=int, default=10000)
    parser.add_argument('--n_subsamples_lsdd', type=int, default=10000)
    parser.add_argument('--n_subsamples_cvm', type=int, default=10000)
    parser.add_argument('--n_subsamples_ks', type=int, default=10000)
    parser.add_argument('--run_driftlens', action='store_true')
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

    print("Model name: ", args.model_name)
    print("Number of runs: ", args.number_of_runs)
    print("Window size: ", args.window_size)
    print("Number of windows: ", args.number_of_windows)
    print("Number of window reference samples : ", args.n_subsamples_mmd)
    print("Number of window reference samples : ", args.n_subsamples_lsdd)
    print("Number of window reference samples : ", args.n_subsamples_cvm)
    print("Number of window reference samples : ", args.n_subsamples_ks)
    print("Number of samples threshold: ", args.threshold_number_of_estimation_samples)
    print("Drift percentage: ", args.drift_percentage)

    training_label_list = [0, 1, 2]  # Labels used for training
    drift_label_list = [3]  # Labels used for drift simulation

    if args.save_results:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        ts = time.time()
        timestamp = str(datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S'))
        output_filename = f"drift_detection_accuracy_model_{args.model_name}_win_size_{args.window_size}_n_windows_{args.number_of_windows}_{timestamp}.json"

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

    print("Training samples:", len(E_train))
    print("Test samples:", len(E_test))
    print("New unseen samples:", len(E_new_unseen))
    print("Drift samples:", len(E_drift))

    ks_acc_dict = {str(p): [] for p in args.drift_percentage}
    mmd_acc_dict = {str(p): [] for p in args.drift_percentage}
    lsdd_acc_dict = {str(p): [] for p in args.drift_percentage}
    cvm_acc_dict = {str(p): [] for p in args.drift_percentage}
    driftlens_acc_dict = {str(p): [] for p in args.drift_percentage}

    output_dict_run_list = []

    output_dict = {"params": vars(args)}

    for run_id in range(args.number_of_runs):

        print(f"\nRun {run_id + 1}/{args.number_of_runs}")

        # Initialize the WindowsGenerator - used for creating the windows
        wg = WindowsGenerator(training_label_list,
                              drift_label_list,
                              E_new_unseen,
                              Y_predicted_new_unseen,
                              Y_original_new_unseen,
                              E_drift,
                              Y_predicted_drift,
                              Y_original_drift)

        if args.run_driftlens:

            # Initialize the DriftLens
            dl = DriftLens()

            # Estimate the baseline with DriftLens
            baseline = dl.estimate_baseline(E=E_train,
                                            Y=Y_predicted_train,
                                            label_list=training_label_list,
                                            batch_n_pc=batch_n_pc,
                                            per_label_n_pc=per_label_n_pc)

            # Estimate the threshold values with DriftLens
            per_batch_distances_sorted, per_label_distances_sorted = dl.random_sampling_threshold_estimation(
                label_list=training_label_list,
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
            left_tail = args.threshold_sensitivity / 100
            right_tail = (100 - args.threshold_sensitivity) / 100
            l = l[(l > np.quantile(l, left_tail)) & (l < np.quantile(l, right_tail))].tolist()
            per_batch_th = max(l)
        else:
            print("DriftLens skipped")

        if (args.n_subsamples_mmd < len(E_train)) and (args.n_subsamples_mmd != -1):

            E_subsample_mmd, Y_subsample_mmd = stratified_subsampling(E_train,
                                                              Y_original_train,
                                                              n_samples=args.n_subsamples_mmd,
                                                              unique_labels=training_label_list)
        else:
            E_subsample_mmd, Y_subsample_mmd = (E_train, Y_original_train)

        if (args.n_subsamples_lsdd < len(E_train)) and (args.n_subsamples_lsdd != -1):

            E_subsample_lsdd, Y_subsample_lsdd = stratified_subsampling(E_train,
                                                              Y_original_train,
                                                              n_samples=args.n_subsamples_lsdd,
                                                              unique_labels=training_label_list)
        else:
            E_subsample_lsdd, Y_subsample_lsdd = (E_train, Y_original_train)

        if (args.n_subsamples_cvm < len(E_train)) and (args.n_subsamples_cvm != -1):

            E_subsample_cvm, Y_subsample_cvm = stratified_subsampling(E_train,
                                                              Y_original_train,
                                                              n_samples=args.n_subsamples_cvm,
                                                              unique_labels=training_label_list)
        else:
            E_subsample_cvm, Y_subsample_cvm = (E_train, Y_original_train)

        if (args.n_subsamples_ks < len(E_train)) and (args.n_subsamples_ks != -1):

            E_subsample_ks, Y_subsample_ks = stratified_subsampling(E_train,
                                                              Y_original_train,
                                                              n_samples=args.n_subsamples_ks,
                                                              unique_labels=training_label_list)
        else:
            E_subsample_ks = E_train
            Y_subsample_ks = Y_original_train

        print(f"MMD reference window size: {E_subsample_mmd.shape[0]}")
        print(f"LSDD reference window size: {E_subsample_lsdd.shape[0]}")
        print(f"CVM reference window size: {E_subsample_cvm.shape[0]}")
        print(f"KS reference window size: {E_subsample_ks.shape[0]}")

        # Initialize drift detectors used for comparison
        ks_detector = KSDrift(E_subsample_ks, p_val=.05)
        mmd_detector = MMDDrift(E_subsample_mmd, p_val=.05, n_permutations=100, backend="pytorch")
        lsdd_detector = LSDDDrift(E_subsample_lsdd, backend='pytorch', p_val=.05)
        cvm_detector = CVMDrift(E_subsample_cvm, p_val=.05)

        for current_drift_percentage in args.drift_percentage:

            print(f" Drift percentage: {current_drift_percentage}")

            # Initialize empty lists of predictions
            ks_preds = []
            mmd_preds = []
            lsdd_preds = []
            cvm_preds = []
            dl_distances = []

            # Ground truth
            if current_drift_percentage > 0:
                ground_truth = [1] * n_windows
            else:
                ground_truth = [0] * n_windows

            # Generate windows and predict drift
            for i in tqdm(range(n_windows)):

                if current_drift_percentage > 0:
                    # Drift
                    E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_constant_drift_windows_generation(window_size=window_size,
                                                                                                                        n_windows=1,
                                                                                                                        drift_percentage=float(current_drift_percentage/100),
                                                                                                                        flag_shuffle=True,
                                                                                                                        flag_replacement=True)

                else:
                    # No Drift
                    E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_without_drift_windows_generation(window_size=window_size,
                                                                                                                      n_windows=1,
                                                                                                                      flag_shuffle=True,
                                                                                                                      flag_replacement=True)

                if args.run_driftlens:
                    # Compute the window distribution distances (Frechet Inception Distance) with DriftLens
                    dl_distance = dl.compute_window_distribution_distances(E_windows[0], Y_predicted_windows[0])

                # Predict drift with the drift detectors used for comparison
                ks_pred = ks_detector.predict(E_windows[0])
                mmd_pred = mmd_detector.predict(E_windows[0])
                lsdd_pred = lsdd_detector.predict(E_windows[0], return_p_val=True, return_distance=True)
                cvm_pred = cvm_detector.predict(E_windows[0], drift_type='batch', return_p_val=True, return_distance=True)

                # Append the predictions to the lists
                ks_preds.append(ks_pred["data"]["is_drift"])
                mmd_preds.append(mmd_pred["data"]["is_drift"])
                lsdd_preds.append(lsdd_pred["data"]["is_drift"])
                cvm_preds.append(cvm_pred["data"]["is_drift"])

                if args.run_driftlens:
                    dl_distances.append(dl_distance)

            if args.run_driftlens:
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

            if args.run_driftlens:
                driftlens_acc = accuracy_score(ground_truth, dl_preds, normalize=True)
            else:
                driftlens_acc = -1

            ks_acc_dict[str(current_drift_percentage)].append(ks_acc)
            mmd_acc_dict[str(current_drift_percentage)].append(mmd_acc)
            lsdd_acc_dict[str(current_drift_percentage)].append(lsdd_acc)
            cvm_acc_dict[str(current_drift_percentage)].append(cvm_acc)
            driftlens_acc_dict[str(current_drift_percentage)].append(driftlens_acc)

            print("MMD: ", mmd_acc)
            print("KS: ", ks_acc)
            print("LSDD: ", lsdd_acc)
            print("CVM: ", cvm_acc)
            print("DriftLens: ", driftlens_acc)

            # Create the output dictionary
            output_dict_run = {f"run_id":run_id, "drift_percentage": current_drift_percentage, "KS": ks_acc, "MMD": mmd_acc, "LSDD": lsdd_acc, "CVM": cvm_acc, "DriftLens": driftlens_acc}
            output_dict_run_list.append(output_dict_run)

    output_dict["runs_log"] = output_dict_run_list

    for p in args.drift_percentage:
        output_dict[str(p)] = {"accuracy_list": {"KS": ks_acc_dict[str(p)], "MMD": mmd_acc_dict[str(p)], "LSDD": lsdd_acc_dict[str(p)], "CVM": cvm_acc_dict[str(p)], "DriftLens": driftlens_acc_dict[str(p)]},
                                  "mean_accuracy": {"KS": np.mean(ks_acc_dict[str(p)]), "MMD": np.mean(mmd_acc_dict[str(p)]), "LSDD": np.mean(lsdd_acc_dict[str(p)]), "CVM": np.mean(cvm_acc_dict[str(p)]), "DriftLens": np.mean(driftlens_acc_dict[str(p)])},
                                  "standard_deviation_accuracy": {"KS": np.std(ks_acc_dict[str(p)]), "MMD": np.std(mmd_acc_dict[str(p)]), "LSDD": np.std(lsdd_acc_dict[str(p)]), "CVM": np.std(cvm_acc_dict[str(p)]), "DriftLens": np.std(driftlens_acc_dict[str(p)])}}

    mean_accuracy_drift_list_mmd = []
    mean_accuracy_drift_list_ks = []
    mean_accuracy_drift_list_lsdd = []
    mean_accuracy_drift_list_cvm = []
    mean_accuracy_drift_list_driftlens = []
    for p in args.drift_percentage:
        if p > 0:
            mean_accuracy_drift_list_mmd.append(output_dict[str(p)]["mean_accuracy"]["MMD"])
            mean_accuracy_drift_list_ks.append(output_dict[str(p)]["mean_accuracy"]["KS"])
            mean_accuracy_drift_list_lsdd.append(output_dict[str(p)]["mean_accuracy"]["LSDD"])
            mean_accuracy_drift_list_cvm.append(output_dict[str(p)]["mean_accuracy"]["CVM"])
            mean_accuracy_drift_list_driftlens.append(output_dict[str(p)]["mean_accuracy"]["DriftLens"])

    hdd_mmd = statistics.harmonic_mean(
        [output_dict['0']["mean_accuracy"]["MMD"], np.mean(mean_accuracy_drift_list_mmd)])
    hdd_ks = statistics.harmonic_mean(
        [output_dict['0']["mean_accuracy"]["KS"], np.mean(mean_accuracy_drift_list_ks)])
    hdd_lsdd = statistics.harmonic_mean(
        [output_dict['0']["mean_accuracy"]["LSDD"], np.mean(mean_accuracy_drift_list_lsdd)])
    hdd_cvm = statistics.harmonic_mean(
        [output_dict['0']["mean_accuracy"]["CVM"], np.mean(mean_accuracy_drift_list_cvm)])

    if args.run_driftlens:
        hdd_driftlens = statistics.harmonic_mean(
            [output_dict['0']["mean_accuracy"]["DriftLens"], np.mean(mean_accuracy_drift_list_driftlens)])
    else:
        hdd_driftlens = -1

    output_dict["HDD"] = {"MMD": hdd_mmd, "KS": hdd_ks, "LSDD": hdd_lsdd, "CVM": hdd_cvm,
                          "DriftLens": hdd_driftlens}
    # Save the output dictionary
    with open(os.path.join(args.output_dir, output_filename), 'w') as fp:
        json.dump(output_dict, fp)

    return


if __name__ == "__main__":
    main()