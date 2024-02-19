import argparse
from experiments.windows_manager.windows_generator import WindowsGenerator
import os
import json
import datetime
import time
from driftlens.driftlens import DriftLens
import h5py
from scipy import stats
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--training_label_list', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--drift_label_list', type=int, nargs='+', default=[3])
    parser.add_argument('--number_of_runs', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='bert')
    parser.add_argument('--window_size', type=int, default=1000)
    parser.add_argument('--number_of_windows', type=int, default=100)
    parser.add_argument('--sudden_drift_offset', type=int, default=50)
    parser.add_argument('--sudden_drift_percentage', type=float, default=0.25)
    parser.add_argument('--incremental_starting_drift_percentage', type=float, default=0.1)
    parser.add_argument('--incremental_drift_increase_rate', type=float, default=0.01)
    parser.add_argument('--incremental_drift_offset', type=float, default=20)
    parser.add_argument('--periodic_drift_percentage', type=float, default=0.4)
    parser.add_argument('--periodic_drift_offset', type=int, default=20)
    parser.add_argument('--periodic_drift_duration', type=int, default=20)
    parser.add_argument('--batch_n_pc', type=int, default=150)
    parser.add_argument('--per_label_n_pc', type=int, default=75)
    parser.add_argument('--train_embedding_filepath', type=str, default=f"{os.getcwd()}/static/saved_embeddings/bert/train_embedding_0_1_2.hdf5")
    parser.add_argument('--test_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/bert/test_embedding_0_1_2.hdf5')
    parser.add_argument('--new_unseen_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/bert/new_unseen_embedding_0_1_2.hdf5')
    parser.add_argument('--drift_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/bert/drift_embedding_3.hdf5')
    parser.add_argument('--output_dir', type=str, default=f"{os.getcwd()}/static/outputs/bert/")
    parser.add_argument('--save_results', action='store_true')
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

def main():
    print("Drift Curve Correlation - Use Case 1")

    # Parse arguments
    args = parse_args()

    print("Model name: ", args.model_name)
    print("Number of runs: ", args.number_of_runs)
    print("Window size: ", args.window_size)
    print("Number of windows: ", args.number_of_windows)

    training_label_list = args.training_label_list  # Labels used for training
    drift_label_list = args.drift_label_list  # Labels used for drift simulation


    if args.save_results:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        ts = time.time()
        timestamp = str(datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S'))
        output_filename = f"drift_correlation_model_{args.model_name}_win_size_{args.window_size}_n_windows_{args.number_of_windows}_{timestamp}.json"

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

    print("Training samples:", len(E_train))
    print("Test samples:", len(E_test))
    print("New unseen samples:", len(E_new_unseen))
    print("Drift samples:", len(E_drift))
    print("\n")

    output_dict = {}

    """
    no_drift_correlation_values = []

    print("No Drift")
    for i in range(args.number_of_runs):
        print(f"Run {i+1}/{args.number_of_runs}")
        E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_without_drift_windows_generation(window_size=args.window_size,
                                                                                                            n_windows=args.number_of_windows,
                                                                                                            flag_shuffle=True,
                                                                                                            flag_replacement=True)
        no_drift_gt = [0]*args.number_of_windows

        distances_dict = dl.compute_window_list_distribution_distances(E_windows, Y_predicted_windows)
        per_batch_distances = [d["per-batch"] for d in distances_dict[0]]

        res = stats.spearmanr(per_batch_distances, no_drift_gt)
        spearmanr_correlation = res.statistic
        no_drift_correlation_values.append(spearmanr_correlation)

    print(no_drift_correlation_values)
    print("Mean Spearman correlation:", np.mean(no_drift_correlation_values))
    print("Std Spearman correlation:", np.std(no_drift_correlation_values))
    print("\n\n")
    """

    sudden_drift_correlation_values = []

    print("Sudden Drift")
    for i in tqdm(range(args.number_of_runs)):
        #print(f"Run {i+1}/{args.number_of_runs}")
        E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_incremental_drift_windows_generation(window_size=args.window_size,
                                                                                                                n_windows=args.number_of_windows,
                                                                                                                starting_drift_percentage=args.sudden_drift_percentage,
                                                                                                                drift_increase_rate=0,
                                                                                                                drift_offset=args.sudden_drift_offset,
                                                                                                                flag_shuffle=True,
                                                                                                                flag_replacement=True)

        sudden_drift_gt = [0] * args.sudden_drift_offset + [args.sudden_drift_percentage] * (args.number_of_windows - args.sudden_drift_offset)

        distances_dict = dl.compute_window_list_distribution_distances(E_windows, Y_predicted_windows)
        per_batch_distances = [d["per-batch"] for d in distances_dict[0]]

        res = stats.spearmanr(per_batch_distances, sudden_drift_gt)
        spearmanr_correlation = res.statistic
        sudden_drift_correlation_values.append(spearmanr_correlation)

    print(sudden_drift_correlation_values)
    print("Mean Spearman correlation:", np.mean(sudden_drift_correlation_values))
    print("Std Spearman correlation:", np.std(sudden_drift_correlation_values))
    print("\n")

    incremental_drift_correlation_values = []
    print("Incremental Drift")
    for i in tqdm(range(args.number_of_runs)):
        #print(f"Run {i+1}/{args.number_of_runs}")
        E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_incremental_drift_windows_generation(window_size=args.window_size,
                                                                                                                n_windows=args.number_of_windows,
                                                                                                                starting_drift_percentage=args.incremental_starting_drift_percentage,
                                                                                                                drift_increase_rate=args.incremental_drift_increase_rate,
                                                                                                                drift_offset=args.incremental_drift_offset,
                                                                                                                flag_shuffle=True,
                                                                                                                flag_replacement=True)

        distances_dict = dl.compute_window_list_distribution_distances(E_windows, Y_predicted_windows)
        per_batch_distances = [d["per-batch"] for d in distances_dict[0]]

        incremental_gt_drift_percentage = 0

        incremental_gt = []
        for j in range(args.number_of_windows):

            if j < args.incremental_drift_offset:
                incremental_gt_drift_percentage = 0

            if j == args.incremental_drift_offset:
                incremental_gt_drift_percentage = args.incremental_starting_drift_percentage

            if j > args.incremental_drift_offset:
                incremental_gt_drift_percentage += args.incremental_drift_increase_rate
                incremental_gt_drift_percentage = float(min(incremental_gt_drift_percentage, 1.0))

            incremental_gt.append(incremental_gt_drift_percentage)

        res = stats.spearmanr(per_batch_distances, incremental_gt)
        spearmanr_correlation = res.statistic
        incremental_drift_correlation_values.append(spearmanr_correlation)

    print(incremental_drift_correlation_values)
    print("Mean Spearman correlation:", np.mean(incremental_drift_correlation_values))
    print("Std Spearman correlation:", np.std(incremental_drift_correlation_values))
    print("\n")

    print("Periodic Drift")
    periodic_drift_correlation_values = []
    for i in tqdm(range(args.number_of_runs)):
        #print(f"Run {i+1}/{args.number_of_runs}")
        E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_periodic_drift_windows_generation(window_size=args.window_size,
                                                                                                                n_windows=args.number_of_windows,
                                                                                                                drift_offset=args.periodic_drift_offset,
                                                                                                                drift_duration=args.periodic_drift_duration,
                                                                                                                drift_percentage=args.periodic_drift_percentage,
                                                                                                                flag_shuffle=True,
                                                                                                                flag_replacement=True)


        distances_dict = dl.compute_window_list_distribution_distances(E_windows, Y_predicted_windows)
        per_batch_distances = [d["per-batch"] for d in distances_dict[0]]


        n_periodic = n_windows // (args.periodic_drift_offset + args.periodic_drift_duration)
        n_periodic_remainder = n_windows % (args.periodic_drift_offset + args.periodic_drift_duration)

        single_periodic_gt = [0] * args.periodic_drift_offset + [args.periodic_drift_percentage] * args.periodic_drift_duration

        periodic_gt = single_periodic_gt * n_periodic

        if n_periodic_remainder > 0:
            if n_periodic_remainder < args.periodic_drift_offset:
                periodic_gt += [0] * n_periodic_remainder
            else:
                periodic_gt += [0] * args.periodic_drift_offset
                periodic_gt += [args.periodic_drift_percentage] * (n_periodic_remainder - args.periodic_drift_offset)

        res = stats.spearmanr(per_batch_distances, periodic_gt)
        spearmanr_correlation = res.statistic
        periodic_drift_correlation_values.append(spearmanr_correlation)

    print(periodic_drift_correlation_values)
    print("Mean Spearman correlation:", np.mean(periodic_drift_correlation_values))
    print("Std Spearman correlation:", np.std(periodic_drift_correlation_values))
    print("\n")

    return

if __name__ == "__main__":
    main()