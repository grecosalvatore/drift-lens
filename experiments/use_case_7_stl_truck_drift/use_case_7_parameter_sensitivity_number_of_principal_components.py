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
    parser.add_argument('--drift_percentage', type=int, nargs='+', default=[0, 5, 10, 15, 20]),
    parser.add_argument('--threshold_number_of_estimation_samples', type=int, default=10000)
    parser.add_argument('--batch_n_pc_list', type=int, nargs='+', default=[50, 100, 150, 200, 250]),
    #parser.add_argument('--batch_n_pc', type=int, default=150)
    parser.add_argument('--per_label_n_pc', type=int, default=75)
    parser.add_argument('--threshold_sensitivity', type=int, default=99)
    parser.add_argument('--train_embedding_filepath', type=str, default=f"{os.getcwd()}/static/saved_embeddings/vit/train_embedding.hdf5")
    parser.add_argument('--test_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/vit/test_embedding.hdf5')
    parser.add_argument('--new_unseen_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/vit/new_unseen_embedding.hdf5')
    parser.add_argument('--drift_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/vit/drift_embedding.hdf5')
    parser.add_argument('--output_dir', type=str, default=f"{os.getcwd()}/static/outputs/vit/")
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
    print("Drift Detection Experiment - Use Case 7")

    # Parse arguments
    args = parse_args()

    print("Model name: ", args.model_name)
    print("Number of runs: ", args.number_of_runs)
    print("Window size: ", args.window_size)
    print("Number of windows: ", args.number_of_windows)
    print("Number of principal componenets sampling: ", args.batch_n_pc_list)
    print("Number of threshold samples: ", args.threshold_number_of_estimation_samples)
    print("Drift percentage: ", args.drift_percentage)

    training_label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Labels used for training
    drift_label_list = [9]  # Labels used for drift simulation

    if args.save_results:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        ts = time.time()
        timestamp = str(datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S'))
        output_filename = f"parameter_sensitivity_number_of_principal_components_{args.model_name}_win_size_{args.window_size}_n_windows_{args.number_of_windows}_{timestamp}.json"

    # Parse parameters
    window_size = args.window_size
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

    output_dict_run_list = []

    output_dict = {"params": vars(args)}

    for batch_n_pc in args.batch_n_pc_list:
        print(f"\nBatch Number of principal components: {batch_n_pc}")

        output_dict[batch_n_pc] = {}

        driftlens_acc_dict = {str(p): [] for p in args.drift_percentage}

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
            l = l[(l > np.quantile(l, 0.01)) & (l < np.quantile(l, 0.99))].tolist()
            per_batch_th = max(l)


            for current_drift_percentage in args.drift_percentage:

                print(f" Drift percentage: {current_drift_percentage}")

                # Initialize empty lists of predictions
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

                    # Compute the window distribution distances (Frechet Inception Distance) with DriftLens
                    dl_distance = dl.compute_window_distribution_distances(E_windows[0], Y_predicted_windows[0])

                    dl_distances.append(dl_distance)


                dl_preds = []
                for dl_distance in dl_distances:
                    if dl_distance["per-batch"] > per_batch_th:
                        dl_preds.append(1)
                    else:
                        dl_preds.append(0)

                # Calculate the accuracy of the drift detectors
                driftlens_acc = accuracy_score(ground_truth, dl_preds, normalize=True)

                driftlens_acc_dict[str(current_drift_percentage)].append(driftlens_acc)

                print("DriftLens: ", driftlens_acc)

                # Create the output dictionary
                output_dict_run = {f"run_id":run_id, "drift_percentage": current_drift_percentage, "batch_n_pc":batch_n_pc, "DriftLens": driftlens_acc}
                output_dict_run_list.append(output_dict_run)

        output_dict["runs_log"] = output_dict_run_list

        for p in args.drift_percentage:
            output_dict[batch_n_pc][str(p)] = {"accuracy_list": {"DriftLens": driftlens_acc_dict[str(p)]},
                                      "mean_accuracy": {"DriftLens": np.mean(driftlens_acc_dict[str(p)])},
                                      "standard_deviation_accuracy": {"DriftLens": np.std(driftlens_acc_dict[str(p)])}}


        # Save the output dictionary
        with open(os.path.join(args.output_dir, output_filename), 'w') as fp:
            json.dump(output_dict, fp)

    return


if __name__ == "__main__":
    main()