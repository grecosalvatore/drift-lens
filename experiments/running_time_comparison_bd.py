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

import argparse

def range_type(astr, nargs=None):
    # Convert the input string into a range
    values = astr.split(":")
    if len(values) == 1:  # Single value
        # Include the single value by setting start and end as the same, adding 1 to include the end
        return range(int(values[0]), int(values[0]) + 1)
    elif len(values) == 2:  # Start and end
        # Add 1 to the end value to include it in the range
        return range(int(values[0]), int(values[1]) + 1)
    elif len(values) == 3:  # Start, end, and step
        # Calculate the adjusted end to potentially include the end value, depending on the step
        start, end, step = int(values[0]), int(values[1]), int(values[2])
        adjusted_end = end + 1 if (end - start) % step == 0 else end
        return range(start, adjusted_end, step)
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

def run_window_drift_prediction(E_window, Y_window, ks_detector, mmd_detector, lsdd_detector, cvm_detector, drift_lens_detector, flag_mmd=True, flag_ks=True, flag_lsdd=True, flag_cvm=True, flag_driftlens=True):

    # Measure the running time of drift_lens_detector.predict
    if flag_driftlens:
        start_time = time.time()
        dl_distance = drift_lens_detector.compute_window_distribution_distances(E_window, Y_window, distribution_distance_metric="bhattacharyya_drift_distance")
        end_time = time.time()
        dl_time = end_time - start_time
    else:
        dl_time = -1
    #print(f"DriftLens Detector Prediction Time: {dl_time} seconds")

    # Measure the running time of ks_detector.predict
    if flag_ks:
        start_time = time.time()
        ks_pred = ks_detector.predict(E_window)
        end_time = time.time()
        ks_time = end_time - start_time
    else:
        ks_time = -1
    #print(f"KS Detector Prediction Time: {ks_time} seconds")

    # Measure the running time of mmd_detector.predict
    if flag_mmd:
        start_time = time.time()
        mmd_pred = mmd_detector.predict(E_window)
        end_time = time.time()
        mmd_time = end_time - start_time
        #print(f"MMD Detector Prediction Time: {mmd_time} seconds")
    else:
        mmd_time = -1

    # Measure the running time of lsdd_detector.predict
    if flag_lsdd:
        start_time = time.time()
        lsdd_pred = lsdd_detector.predict(E_window, return_p_val=True, return_distance=True)
        end_time = time.time()
        lsdd_time = end_time - start_time
        #print(f"LSDD Detector Prediction Time: {lsdd_time} seconds")
    else:
        lsdd_time = -1
    #print(f"LSDD Detector Prediction Time: {lsdd_time} seconds")

    # Measure the running time of cvm_detector.predict
    if flag_cvm:
        start_time = time.time()
        cvm_pred = cvm_detector.predict(E_window, drift_type='batch', return_p_val=True, return_distance=True)
        end_time = time.time()
        cvm_time = end_time - start_time
        #print(f"CVM Detector Prediction Time: {cvm_time} seconds")
    else:
        cvm_time = -1
    #print(f"CVM Detector Prediction Time: {cvm_time} seconds")

    running_time_dict = {"DriftLens": dl_time, "KS": ks_time, "MMD": mmd_time, "LSDD": lsdd_time, "CVM": cvm_time}
    return running_time_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Running Time Comparison')
    parser.add_argument('--number_of_runs', type=int, default=10)
    parser.add_argument('--training_label_list', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--drift_label_list', type=int, nargs='+', default=[3])
    parser.add_argument('--reference_window_size_range', type=range_type, default="500:1500:500", help='Example range argument in start:end:step format')
    parser.add_argument('--datastream_window_size_range', type=range_type, default="500:1500:500", help='Example range argument in start:end:step format')
    parser.add_argument('--embedding_dimensionality_range', type=range_type, default="500:1500:500", help='Example range argument in start:end:step format')
    parser.add_argument('--run_mmd', action='store_true')
    parser.add_argument('--run_ks', action='store_true')
    parser.add_argument('--run_lsdd', action='store_true')
    parser.add_argument('--run_cvm', action='store_true')
    parser.add_argument('--run_driftlens', action='store_true')
    parser.add_argument('--run_experiment_reference_window', action='store_true')
    parser.add_argument('--run_experiment_window_size', action='store_true')
    parser.add_argument('--run_experiment_embedding_dimensionality', action='store_true')
    parser.add_argument('--fixed_reference_window_size', type=int, default=1000)
    parser.add_argument('--fixed_datastream_window_size', type=int, default=1000)
    parser.add_argument('--fixed_embedding_dimensionality', type=int, default=1000)
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

    detectors = ["DriftLens", "KS", "MMD", "LSDD", "CVM"]

    running_time_for_reference_window_size_dict = {}
    if args.run_experiment_reference_window:

        E_window_datastream_fixed = np.random.uniform(low=-2, high=2, size=(args.fixed_datastream_window_size,
                                                                            args.fixed_embedding_dimensionality))

        Y_window_datastream_fixed = np.random.randint(low=0, high=num_labels, size=args.fixed_datastream_window_size)

        print("Computing running time comparison based on the reference window size")
        # Running time comparison based on the reference window size
        for reference_window_size in tqdm(args.reference_window_size_range):
            print("Current reference window size", reference_window_size)

            running_time_for_reference_window_size_dict[reference_window_size] = {}
            for drift_detector in detectors:
                running_time_for_reference_window_size_dict[reference_window_size][drift_detector] = {}
                running_time_for_reference_window_size_dict[reference_window_size][drift_detector]['running_time_list'] = []

            E_reference = np.random.uniform(low=-2, high=2, size=(reference_window_size,
                                                                  args.fixed_embedding_dimensionality))

            Y_reference = np.random.randint(low=0, high=num_labels, size=reference_window_size)

            if args.run_driftlens:
                drift_lens_detector = DriftLens()
                baseline = drift_lens_detector.estimate_baseline(E=E_reference,
                                                                 Y=Y_reference,
                                                                 label_list=training_label_list,
                                                                 batch_n_pc=args.batch_n_pc,
                                                                 per_label_n_pc=args.per_label_n_pc)
            else:
                drift_lens_detector = None

            if args.run_ks:
                ks_detector = KSDrift(E_reference, p_val=.05)
            else:
                ks_detector = None

            if args.run_mmd:
                mmd_detector = MMDDrift(E_reference, p_val=.05, n_permutations=100, backend="pytorch")
            else:
                mmd_detector = None

            if args.run_lsdd:
                lsdd_detector = LSDDDrift(E_reference, backend='pytorch', p_val=.05)
            else:
                lsdd_detector = None

            if args.run_cvm:
                cvm_detector = CVMDrift(E_reference, p_val=.05)
            else:
                cvm_detector = None

            for i in tqdm(range(args.number_of_runs)):
                running_time_dict = run_window_drift_prediction(E_window_datastream_fixed,
                                                                Y_window_datastream_fixed,
                                                                ks_detector,
                                                                mmd_detector,
                                                                lsdd_detector,
                                                                cvm_detector,
                                                                drift_lens_detector,
                                                                flag_mmd=args.run_mmd,
                                                                flag_ks=args.run_ks,
                                                                flag_lsdd=args.run_lsdd,
                                                                flag_cvm=args.run_cvm,
                                                                flag_driftlens=args.run_driftlens)


                for key in running_time_dict:
                    running_time_for_reference_window_size_dict[reference_window_size][key]['running_time_list'].append(running_time_dict[key])


            for key in detectors:
                running_time_for_reference_window_size_dict[reference_window_size][key]['mean'] = np.mean(running_time_for_reference_window_size_dict[reference_window_size][key]['running_time_list'])
                running_time_for_reference_window_size_dict[reference_window_size][key]['std'] = np.std(running_time_for_reference_window_size_dict[reference_window_size][key]['running_time_list'])


        print(running_time_for_reference_window_size_dict)
        print("\n\n")

    ##################################################################################################
    # Running time comparison based on the datastream window size

    running_time_for_datastream_window_size_dict = {}

    if args.run_experiment_window_size:
        print("Computing running time comparison based on the datastream window size")



        E_reference_fixed = np.random.uniform(low=-2, high=2, size=(args.fixed_reference_window_size,
                                                                    args.fixed_embedding_dimensionality))

        Y_reference_fixed = np.random.randint(low=0, high=num_labels, size=args.fixed_reference_window_size)

        if args.run_driftlens:
            drift_lens_detector = DriftLens()
            baseline = drift_lens_detector.estimate_baseline(E=E_reference_fixed,
                                                             Y=Y_reference_fixed,
                                                             label_list=training_label_list,
                                                             batch_n_pc=args.batch_n_pc,
                                                             per_label_n_pc=args.per_label_n_pc)
        else:
            drift_lens_detector = None

        if args.run_ks:
            ks_detector = KSDrift(E_reference_fixed, p_val=.05)
        else:
            ks_detector = None

        if args.run_mmd:
            mmd_detector = MMDDrift(E_reference_fixed, p_val=.05, n_permutations=100, backend="pytorch")
        else:
            mmd_detector = None

        if args.run_lsdd:
            lsdd_detector = LSDDDrift(E_reference_fixed, backend='pytorch', p_val=.05)
        else:
            lsdd_detector = None

        if args.run_cvm:
            cvm_detector = CVMDrift(E_reference_fixed, p_val=.05)
        else:
            cvm_detector = None

        running_time_for_datastream_window_size_dict = {}

        # Running time comparison based on the datastream window size
        for datastream_window_size in args.datastream_window_size_range:
            print("Current datastream window size", datastream_window_size)

            running_time_for_datastream_window_size_dict[datastream_window_size] = {}
            for drift_detector in detectors:
                running_time_for_datastream_window_size_dict[datastream_window_size][drift_detector] = {}
                running_time_for_datastream_window_size_dict[datastream_window_size][drift_detector]['running_time_list'] = []

            E_window_datastream = np.random.uniform(low=-2, high=2, size=(datastream_window_size, args.fixed_embedding_dimensionality))

            Y_window_datastream = np.random.randint(low=0, high=num_labels, size=datastream_window_size)

            for i in range(args.number_of_runs):

                running_time_dict = run_window_drift_prediction(E_window_datastream,
                                                                Y_window_datastream,
                                                                ks_detector,
                                                                mmd_detector,
                                                                lsdd_detector,
                                                                cvm_detector,
                                                                drift_lens_detector,
                                                                flag_mmd=args.run_mmd,
                                                                flag_ks=args.run_ks,
                                                                flag_lsdd=args.run_lsdd,
                                                                flag_cvm=args.run_cvm,
                                                                flag_driftlens=args.run_driftlens)

                for key in running_time_dict:
                    running_time_for_datastream_window_size_dict[datastream_window_size][key]['running_time_list'].append(running_time_dict[key])

            for key in detectors:
                running_time_for_datastream_window_size_dict[datastream_window_size][key]['mean'] = np.mean(
                    running_time_for_datastream_window_size_dict[datastream_window_size][key]['running_time_list'])
                running_time_for_datastream_window_size_dict[datastream_window_size][key]['std'] = np.std(
                    running_time_for_datastream_window_size_dict[datastream_window_size][key]['running_time_list'])


        print(running_time_for_datastream_window_size_dict)
        print("\n\n")

    ##################################################################################################
    # Running time comparison based on the embedding dimensionality window size



    running_time_for_embedding_dimensionality_dict = {}

    if args.run_experiment_embedding_dimensionality:
        print("Computing running time comparison based on the datastream window size")

        # Running time comparison based on the datastream window size
        for embedding_dimensionality in tqdm(args.embedding_dimensionality_range):
            print("Current embedding dimensionality window size", embedding_dimensionality)

            running_time_for_embedding_dimensionality_dict[embedding_dimensionality] = {}
            for drift_detector in detectors:
                running_time_for_embedding_dimensionality_dict[embedding_dimensionality][drift_detector] = {}
                running_time_for_embedding_dimensionality_dict[embedding_dimensionality][drift_detector][
                    'running_time_list'] = []

            E_reference_fixed = np.random.uniform(low=-2, high=2, size=(args.fixed_reference_window_size,
                                                                        embedding_dimensionality))

            Y_reference_fixed = np.random.randint(low=0, high=num_labels, size=args.fixed_reference_window_size)


            if args.run_driftlens:
                # Initialize the DriftLens
                drift_lens_detector = DriftLens()

                # Estimate the baseline with DriftLens
                baseline = drift_lens_detector.estimate_baseline(E=E_reference_fixed,
                                                                 Y=Y_reference_fixed,
                                                                 label_list=training_label_list,
                                                                 batch_n_pc=args.batch_n_pc,
                                                                 per_label_n_pc=args.per_label_n_pc)
            else:
                drift_lens_detector = None

            if args.run_ks:
                ks_detector = KSDrift(E_reference_fixed, p_val=.05)
            else:
                ks_detector = None

            if args.run_mmd:
                mmd_detector = MMDDrift(E_reference_fixed, p_val=.05, n_permutations=100, backend="pytorch")
            else:
                mmd_detector = None

            if args.run_lsdd:
                lsdd_detector = LSDDDrift(E_reference_fixed, backend='pytorch', p_val=.05)
            else:
                lsdd_detector = None

            if args.run_cvm:
                cvm_detector = CVMDrift(E_reference_fixed, p_val=.05)
            else:
                cvm_detector = None

            E_window_datastream = np.random.uniform(low=-2, high=2,
                                                    size=(args.fixed_datastream_window_size, embedding_dimensionality))

            Y_window_datastream = np.random.randint(low=0, high=num_labels, size=args.fixed_datastream_window_size)

            for i in range(args.number_of_runs):

                running_time_dict = run_window_drift_prediction(E_window_datastream,
                                                                Y_window_datastream,
                                                                ks_detector,
                                                                mmd_detector,
                                                                lsdd_detector,
                                                                cvm_detector,
                                                                drift_lens_detector,
                                                                flag_mmd=args.run_mmd,
                                                                flag_ks=args.run_ks,
                                                                flag_lsdd=args.run_lsdd,
                                                                flag_cvm=args.run_cvm,
                                                                flag_driftlens=args.run_driftlens)

                for key in running_time_dict:
                    running_time_for_embedding_dimensionality_dict[embedding_dimensionality][key]['running_time_list'].append(
                        running_time_dict[key])

            for key in detectors:
                running_time_for_embedding_dimensionality_dict[embedding_dimensionality][key]['mean'] = np.mean(
                    running_time_for_embedding_dimensionality_dict[embedding_dimensionality][key]['running_time_list'])
                running_time_for_embedding_dimensionality_dict[embedding_dimensionality][key]['std'] = np.std(
                    running_time_for_embedding_dimensionality_dict[embedding_dimensionality][key]['running_time_list'])

        print(running_time_for_embedding_dimensionality_dict)
        print("\n\n")

    return



if __name__ == '__main__':
    main()