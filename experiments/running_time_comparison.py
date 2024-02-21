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

def run_window_drift_prediction(E_window, Y_window, ks_detector, mmd_detector, lsdd_detector, cvm_detector, drift_lens_detector):

    # Measure the running time of drift_lens_detector.predict
    start_time = time.time()
    dl_distance = drift_lens_detector.compute_window_distribution_distances(E_window, Y_window)
    end_time = time.time()
    dl_time = end_time - start_time
    print(f"DriftLens Detector Prediction Time: {dl_time} seconds")

    # Measure the running time of ks_detector.predict
    start_time = time.time()
    ks_pred = ks_detector.predict(E_window)
    end_time = time.time()
    ks_time = end_time - start_time
    print(f"KS Detector Prediction Time: {ks_time} seconds")

    # Measure the running time of mmd_detector.predict
    start_time = time.time()
    mmd_pred = mmd_detector.predict(E_window)
    end_time = time.time()
    mmd_time = end_time - start_time
    print(f"MMD Detector Prediction Time: {mmd_time} seconds")

    # Measure the running time of lsdd_detector.predict
    start_time = time.time()
    lsdd_pred = lsdd_detector.predict(E_window, return_p_val=True, return_distance=True)
    end_time = time.time()
    lsdd_time = end_time - start_time
    print(f"LSDD Detector Prediction Time: {lsdd_time} seconds")

    # Measure the running time of cvm_detector.predict
    start_time = time.time()
    cvm_pred = cvm_detector.predict(E_window, drift_type='batch', return_p_val=True, return_distance=True)
    end_time = time.time()
    cvm_time = end_time - start_time
    print(f"CVM Detector Prediction Time: {cvm_time} seconds")

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

    # Running time comparison based on the reference window size
    for reference_window_size in args.reference_window_size_range:
        print(reference_window_size)

    # Running time comparison based on the datastream window size
    for datastream_window_size in args.datastream_window_size_range:
        print(datastream_window_size)

    return



if __name__ == '__main__':
    main()