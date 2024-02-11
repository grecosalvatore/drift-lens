import argparse
from alibi_detect.cd import KSDrift, MMDDrift, LSDDDrift
from experiments.windows_manager.windows_generator import WindowsGenerator
from driftlens.driftlens import DriftLens
import os
import h5py
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--model_name', type=str, default='bert')
    parser.add_argument('--window_size', type=int, default=2000)
    parser.add_argument('--number_of_windows', type=int, default=10)
    parser.add_argument('--drift_percentage', type=int, default=20)
    parser.add_argument('--batch_n_pc', type=int, default=150)
    parser.add_argument('--per_label_n_pc', type=int, default=75)
    parser.add_argument('--threshold_sensitivity', type=int, default=99)
    parser.add_argument('--train_embedding_filepath', type=str, default=f"{os.getcwd()}/static/saved_embeddings/bert/train_embedding_0_1_2.hdf5")
    parser.add_argument('--test_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/bert/test_embedding_0_1_2.hdf5')
    parser.add_argument('--new_unseen_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/bert/new_unseen_embedding_0_1_2.hdf5')
    parser.add_argument('--drift_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/bert/drift_embedding_3.hdf5')
    parser.add_argument('--output_dir', type=str, default='outputs/')  # Currently not used
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


def main():
    # Parse arguments
    args = parse_args()

    training_label_list = [0, 1, 2]  # Labels used for training - 0: World, 1: Sports, 2: Business
    drift_label_list = [3]  # Labels used for drift simulation - 3: Science/technology

    # Set the file paths for the embeddings
    #train_embedding_filepath = f"{os.getcwd()}/static/saved_embeddings/{args.model_name}/{args.train_embedding_filename}"
    #test_embedding_filepath = f"{os.getcwd()}/static/saved_embeddings/{args.model_name}/{args.test_embedding_filename}"
    #new_unseen_embedding_filepath = f"{os.getcwd()}/static/saved_embeddings/{args.model_name}/{args.new_unseen_embedding_filename}"
    #drift_embedding_filepath = f"{os.getcwd()}/static/saved_embeddings/{args.model_name}/{args.drift_embedding_filename}"

    # Print the current working directory
    print("Current Working Directory:", os.getcwd())

    print(args.model_name)
    print(args.window_size)
    print(args.train_embedding_filepath)

    window_size = args.window_size
    batch_n_pc = args.batch_n_pc
    per_label_n_pc = args.per_label_n_pc
    n_windows = args.number_of_windows

    # Load the embeddings
    E_train, Y_original_train, Y_predicted_train = load_embedding(args.train_embedding_filepath)
    E_test, Y_original_test, Y_predicted_test = load_embedding(args.test_embedding_filepath)
    E_new_unseen, Y_original_new_unseen, Y_predicted_new_unseen = load_embedding(args.new_unseen_embedding_filepath)
    E_drift, Y_original_drift, Y_predicted_drift = load_embedding(args.drift_embedding_filepath)

    print(len(E_train))
    print(len(E_test))
    print(len(E_new_unseen))
    print(len(E_drift))

    ks_preds = []
    mmd_preds = []
    lsdd_preds = []
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
                                                                                                     n_samples=10,
                                                                                                     flag_shuffle=True,
                                                                                                     flag_replacement=True)

    #print(per_batch_distances_sorted, per_label_distances_sorted)

    per_batch_th = max(per_batch_distances_sorted)

    # Initialize drift detectors used for comparison
    ks_detector = KSDrift(E_train, p_val=.05)
    mmd_detector = MMDDrift(E_test, p_val=.05, n_permutations=100, backend="pytorch")
    lsdd_detector = LSDDDrift(E_test, backend='pytorch', p_val=.05)



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

        # Append the predictions to the lists
        ks_preds.append(ks_pred["data"]["is_drift"])
        mmd_preds.append(mmd_pred["data"]["is_drift"])
        lsdd_preds.append(lsdd_pred["data"]["is_drift"])
        dl_distances.append(dl_distance)

    if args.drift_percentage > 0:
        ground_truth = [1] * n_windows
    else:
        ground_truth = [0] * n_windows

    dl_preds = []
    for dl_distance in dl_distances:
        if dl_distance["per-batch"] > per_batch_th:
            dl_preds.append(1)
        else:
            dl_preds.append(0)

    print("KS", ks_preds)
    print("MMD", mmd_preds)
    print("LSDD", lsdd_preds)
    print("DriftLens", dl_preds)

    return


if __name__ == "__main__":
    main()