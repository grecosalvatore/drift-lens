import argparse
from alibi_detect.cd import KSDrift, MMDDrift, LSDDDrift, CVMDrift
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
import time

def range_type(astr, nargs=None):
    # Convert the input string into a range
    values = astr.split(":")
    if len(values) == 1: # Single value
        return range(int(values[0]), int(values[0]) + 1)
    elif len(values) == 2: # Start and end
        return range(int(values[0]), int(values[1]))
    elif len(values) == 3: # Start, end, and step
        return range(int(values[0]), int(values[1]), int(values[2]))
    else:
        raise argparse.ArgumentTypeError("Range values must be in start:end:step format")

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

def run_window_drift_prediction(E_window, Y_window, ks_detector, mmd_detector, lsdd_detector, cvm_detector, drift_lens_detector):

    # Measure the running time of drift_lens_detector.predict
    start_time = time.time()
    dl_distance = drift_lens_detector.compute_window_distribution_distances(E_window, Y_window)
    end_time = time.time()
    dl_time = end_time - start_time
    #print(f"DriftLens Detector Prediction Time: {dl_time} seconds")

    # Measure the running time of ks_detector.predict
    start_time = time.time()
    ks_pred = ks_detector.predict(E_window)
    end_time = time.time()
    ks_time = end_time - start_time
    #print(f"KS Detector Prediction Time: {ks_time} seconds")

    # Measure the running time of mmd_detector.predict
    start_time = time.time()
    mmd_pred = mmd_detector.predict(E_window)
    end_time = time.time()
    mmd_time = end_time - start_time
    #print(f"MMD Detector Prediction Time: {mmd_time} seconds")

    # Measure the running time of lsdd_detector.predict
    start_time = time.time()
    lsdd_pred = lsdd_detector.predict(E_window, return_p_val=True, return_distance=True)
    end_time = time.time()
    lsdd_time = end_time - start_time
    #print(f"LSDD Detector Prediction Time: {lsdd_time} seconds")

    # Measure the running time of cvm_detector.predict
    start_time = time.time()
    cvm_pred = cvm_detector.predict(E_window, drift_type='batch', return_p_val=True, return_distance=True)
    end_time = time.time()
    cvm_time = end_time - start_time
    #print(f"CVM Detector Prediction Time: {cvm_time} seconds")

    running_time_dict = {"DriftLens": dl_time, "KS": ks_time, "MMD": mmd_time, "LSDD": lsdd_time, "CVM": cvm_time}
    return running_time_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Running Time Comparison')
    parser.add_argument('--number_of_runs', type=int, default=1)
    parser.add_argument('--training_label_list', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--drift_label_list', type=int, nargs='+', default=[3])
    parser.add_argument('--reference_window_size_range', type=range_type, default="5000:50000:5000", help='Example range argument in start:end:step format')
    parser.add_argument('--datastream_window_size_range', type=range_type, default="500:5000:500", help='Example range argument in start:end:step format')
    parser.add_argument('--fixed_reference_window_size', type=int, default=1000)
    parser.add_argument('--fixed_datastream_window_size', type=int, default=1000)
    parser.add_argument('--fixed_embedding_dimensionality', type=int, default=1000)
    parser.add_argument('--model_name', type=str, default='bert')
    parser.add_argument('--train_embedding_filepath', type=str, default=f"{os.getcwd()}/static/saved_embeddings/bert/train_embedding_0_1_2.hdf5")
    parser.add_argument('--test_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/bert/test_embedding_0_1_2.hdf5')
    parser.add_argument('--new_unseen_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/bert/new_unseen_embedding_0_1_2.hdf5')
    parser.add_argument('--drift_embedding_filepath', type=str, default=f'{os.getcwd()}/static/saved_embeddings/bert/drift_embedding_3.hdf5')
    parser.add_argument('--output_dir', type=str, default=f"{os.getcwd()}/static/outputs/bert/")
    parser.add_argument('--batch_n_pc', type=int, default=150)
    parser.add_argument('--per_label_n_pc', type=int, default=75)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()







def main():

    # Parse arguments
    args = parse_args()

    training_label_list = args.training_label_list
    num_labels = len(training_label_list)

    E_window_datastream_fixed = np.random.uniform(low=-2, high=2, size=(args.fixed_datastream_window_size,
                                                                        args.fixed_embedding_dimensionality))

    Y_window_datastream_fixed = np.random.randint(low=0, high=num_labels, size=args.fixed_datastream_window_size)

    print("Computing running time comparison based on the reference window size")

    running_time_for_reference_window_size_dict = {}

    # Running time comparison based on the reference window size
    for reference_window_size in tqdm(args.reference_window_size_range):

        E_reference = np.random.uniform(low=-2, high=2, size=(reference_window_size,
                                                              args.fixed_embedding_dimensionality))

        Y_reference = np.random.randint(low=0, high=num_labels, size=reference_window_size)

        # Initialize the DriftLens
        drift_lens_detector = DriftLens()

        # Estimate the baseline with DriftLens
        baseline = drift_lens_detector.estimate_baseline(E=E_reference,
                                                         Y=Y_reference,
                                                         label_list=training_label_list,
                                                         batch_n_pc=args.batch_n_pc,
                                                         per_label_n_pc=args.per_label_n_pc)

        ks_detector = KSDrift(E_reference, p_val=.05)
        mmd_detector = MMDDrift(E_reference, p_val=.05, n_permutations=100, backend="pytorch")
        lsdd_detector = LSDDDrift(E_reference, backend='pytorch', p_val=.05)
        cvm_detector = CVMDrift(E_reference, p_val=.05)

        running_time_for_reference_window_size_dict_tmp = {"DriftLens": [],
                                                       "KS": [],
                                                       "MMD": [],
                                                       "LSDD": [],
                                                       "CVM": []}

        for i in range(args.number_of_runs):

            running_time_dict = run_window_drift_prediction(E_window_datastream_fixed,
                                                            Y_window_datastream_fixed,
                                                            ks_detector,
                                                            mmd_detector,
                                                            lsdd_detector,
                                                            cvm_detector,
                                                            drift_lens_detector)

            for key in running_time_dict:
                running_time_for_reference_window_size_dict_tmp[key].append(running_time_dict[key])

    print(running_time_for_reference_window_size_dict)

    # Running time comparison based on the datastream window size
    for datastream_window_size in args.datastream_window_size_range:
        print(datastream_window_size)

    return



if __name__ == '__main__':
    main()