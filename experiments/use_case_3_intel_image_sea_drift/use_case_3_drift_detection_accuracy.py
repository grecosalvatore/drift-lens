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


def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--model_name', type=str, default='vgg16')
    parser.add_argument('--window_size', type=int, default=1000)
    parser.add_argument('--number_of_windows', type=int, default=10)
    parser.add_argument('--drift_percentage', type=int, default=0)
    parser.add_argument('--batch_n_pc', type=int, default=150)
    parser.add_argument('--per_label_n_pc', type=int, default=75)
    parser.add_argument('--threshold_sensitivity', type=int, default=99)
    parser.add_argument('--threshold_number_of_estimation_samples', type=int, default=100)
    parser.add_argument('--train_embedding_filepath', type=str, default=f"{os.getcwd()}/static/saved_embeddings/vit/train_embedding.hdf5")
    parser.add_argument('--test_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/vit/test_embedding.hdf5')
    parser.add_argument('--new_unseen_embedding_filepath', type=str, default=f'{os.getcwd()}/static/vit/vgg16/new_unseen_embedding.hdf5')
    parser.add_argument('--drift_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/vit/drift_embedding.hdf5')
    parser.add_argument('--output_dir', type=str, default=f"{os.getcwd()}/static/outputs/vgg16/")
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--sota_comparison', action='store_true')
    #parser.add_argument('--sota_comparison', action='store_false')
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

def create_proportions_dict(Y):
    Y = np.array(Y)
    labels, counts = np.unique(Y, return_counts=True)
    total_count = counts.sum()
    proportions = {str(label): count / total_count for label, count in zip(labels, counts)}
    return proportions


def main():
    print("Drift Detection Experiment - Use Case 3 - Intel Image with one class for drift simulation (Sea)")


    # Parse arguments
    args = parse_args()

    print(vars(args))

    training_label_list = [0, 1, 2, 3, 4]  # Labels used for training
    drift_label_list = [5]  # Labels used for drift simulation

    # Print the current working directory
    print("Current Working Directory:", os.getcwd())

    #if args.save_results:
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

    # Create the proportions dictionary by counting the ocurrences of each label in the training set
    proportions_dict = create_proportions_dict(Y_original_train)

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

    print(proportions_dict)

    #E_th = np.vstack([E_train, E_test])
    #Y_original_th = np.hstack([Y_original_train, Y_original_test])
    #Y_predicted_th = np.hstack([Y_predicted_train, Y_predicted_test])

    E_th = E_test
    Y_th = Y_original_test

    # Estimate the threshold values with DriftLens
    per_batch_distances_sorted, per_label_distances_sorted = dl.random_sampling_threshold_estimation(label_list=training_label_list,
                                                                                                     E=E_th,
                                                                                                     Y=Y_th,
                                                                                                     batch_n_pc=batch_n_pc,
                                                                                                     per_label_n_pc=per_label_n_pc,
                                                                                                     window_size=window_size,
                                                                                                     n_samples=args.threshold_number_of_estimation_samples,
                                                                                                     flag_shuffle=True,
                                                                                                     flag_replacement=True,
                                                                                                     proportional_flag=False,
                                                                                                     proportions_dict=proportions_dict)

    # Calculate the threshold values
    print(per_batch_distances_sorted)

    l = np.array(per_batch_distances_sorted)
    l = l[(l > np.quantile(l, 0.01)) & (l < np.quantile(l, 0.99))].tolist()
    per_batch_th = max(l)

    if args.sota_comparison:
        # Initialize drift detectors used for comparison
        ks_detector = KSDrift(E_train[:1000], p_val=.05)
        mmd_detector = MMDDrift(E_test[:1000], p_val=.05, n_permutations=100, backend="pytorch")
        lsdd_detector = LSDDDrift(E_test[:1000], backend='pytorch', p_val=.05)
        cvm_detector = CVMDrift(E_train[:1000], p_val=.05)
        chisquare_detector = ChiSquareDrift(E_train[:1000], p_val=0.05)

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
        dl_distances.append(dl_distance)

        if args.sota_comparison:
            # Predict drift with the drift detectors used for comparison
            ks_pred = ks_detector.predict(E_windows[0])
            mmd_pred = mmd_detector.predict(E_windows[0])
            lsdd_pred = lsdd_detector.predict(E_windows[0], return_p_val=True, return_distance=True)
            cvm_pred = cvm_detector.predict(E_windows[0], drift_type='batch', return_p_val=True, return_distance=True)
            chisquare_pred = chisquare_detector.predict(E_windows[0], drift_type='batch', return_p_val=True, return_distance=True)

            # Append the predictions to the lists
            ks_preds.append(ks_pred["data"]["is_drift"])
            mmd_preds.append(mmd_pred["data"]["is_drift"])
            lsdd_preds.append(lsdd_pred["data"]["is_drift"])
            cvm_preds.append(cvm_pred["data"]["is_drift"])
            chisquare_preds.append(chisquare_pred["data"]["is_drift"])



    if args.drift_percentage > 0:
        ground_truth = [1] * n_windows
    else:
        ground_truth = [0] * n_windows

    print("DriftLens Th", per_batch_th)
    print("\n")
    print(dl_distances)

    dl_preds = []
    for dl_distance in dl_distances:
        if dl_distance["per-batch"] > per_batch_th:
            dl_preds.append(1)
        else:
            dl_preds.append(0)

    if args.sota_comparison:
        # Calculate the accuracy of the drift detectors
        ks_acc = accuracy_score(ground_truth, ks_preds, normalize=True)
        mmd_acc = accuracy_score(ground_truth, mmd_preds, normalize=True)
        lsdd_acc = accuracy_score(ground_truth, lsdd_preds, normalize=True)
        cvm_acc = accuracy_score(ground_truth, cvm_preds, normalize=True)
        chisquare_acc = accuracy_score(ground_truth, chisquare_preds, normalize=True)

    driftlens_acc = accuracy_score(ground_truth, dl_preds, normalize=True)

    if args.sota_comparison:
        print("KS: ", ks_acc)
        print("MMD: ", mmd_acc)
        print("LSDD: ", lsdd_acc)
        print("CVM: ", cvm_acc)
        print("ChiSquare: ", chisquare_acc)

    print("DriftLens: ", driftlens_acc)

    if args.sota_comparison:
        # Create the output dictionary
        output_dict = {"params": vars(args),
                       "accuracy": {"KS": ks_acc, "MMD": mmd_acc, "LSDD": lsdd_acc, "CVM": cvm_acc, "ChiSquare": chisquare_acc, "DriftLens": driftlens_acc}}
    else:
        output_dict = {"params": vars(args),
                       "accuracy": {"DriftLens": driftlens_acc}}

    # Save the output dictionary
    with open(os.path.join(args.output_dir, output_filename), 'w') as fp:
        json.dump(output_dict, fp)

    return


if __name__ == "__main__":
    main()