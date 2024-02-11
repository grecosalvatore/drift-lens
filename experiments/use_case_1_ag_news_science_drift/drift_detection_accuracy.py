import argparse
from alibi_detect.cd import KSDrift, MMDDrift, LSDDDrift
from experiments.windows_manager.windows_generator import WindowsGenerator
import os
import h5py
'''
------------------------------------------------------------------------------------------------------------------------

                                            PARSING ARGUMENTS

------------------------------------------------------------------------------------------------------------------------
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--model_name', type=str, default='bert')
    parser.add_argument('--window_size', type=int, default=2000)
    parser.add_argument('--drift_percentage', type=float, default=0.20)
    parser.add_argument('--train_embedding_filename', type=str, default='/home/mlaquatra/nlp_projects/geolingit/geolingit-data/standard-track/subtask_b/train_b.tsv')
    parser.add_argument('--test_embedding_filename', type=str, default='/home/mlaquatra/nlp_projects/geolingit/geolingit-data/standard-track/subtask_b/dev_b.tsv')
    parser.add_argument('--new_unseen_embedding_filename', type=str, default='/home/mlaquatra/nlp_projects/geolingit/geolingit-data/standard-track/subtask_b/dev_b.tsv')
    parser.add_argument('--drift_embedding_filename', type=str, default='/home/mlaquatra/nlp_projects/geolingit/geolingit-data/standard-track/subtask_b/dev_b.tsv')
    parser.add_argument('--output_dir', type=str, default='ft_models_b/')
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()

args = parse_args()


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
        raise Exception("Experiment Manager: Error in loading the embedding file. Please set the embedding paths in the configuration file.")
    return E, Y_original, Y_predicted


training_label_list = [0, 1, 2]
drift_label_list = [3]


train_embedding_filepath = f"{os.getcwd()}/experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/{args.model_name}/{args.train_embedding_filename}"
test_embedding_filepath = f"{os.getcwd()}/experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/{args.model_name}/{args.test_embedding_filename}"
new_unseen_embedding_filepath = f"{os.getcwd()}/experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/{args.model_name}/{args.new_unseen_embedding_filename}"
drift_embedding_filepath = f"{os.getcwd()}/experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/{args.model_name}/{args.drift_embedding_filename}"



# Print the current working directory
print("Current Working Directory:", os.getcwd())

print(args.model_name)
print(args.window_size)
print(args.train_embedding_filename)

E_train, Y_original_train, Y_predicted_train = load_embedding(train_embedding_filepath)
E_test, Y_original_test, Y_predicted_test = load_embedding(test_embedding_filepath)
E_new_unseen, Y_original_new_unseen, Y_predicted_new_unseen = load_embedding(new_unseen_embedding_filepath)
E_drift, Y_original_drift, Y_predicted_drift = load_embedding(drift_embedding_filepath)

print(len(E_train))
print(len(E_test))
print(len(E_new_unseen))
print(len(E_drift))

ks_preds = []
mmd_preds = []
lsdd_preds = []


wg = WindowsGenerator(training_label_list,
                        drift_label_list,
                        E_new_unseen,
                        Y_predicted_new_unseen,
                        Y_original_new_unseen,
                        E_drift,
                        Y_predicted_drift,
                        Y_original_drift)

ks_detector = KSDrift(E_train, p_val=.05)
mmd_detector = MMDDrift(E_test, p_val=.05, n_permutations=100, backend="pytorch")
lsdd_detector = LSDDDrift(E_test, backend='pytorch', p_val=.05)

for i in tqdm(range(5)):
    E_windows, Y_predicted_windows, Y_original_windows = wg.balanced_without_drift_windows_generation(
        window_size=window_size,
        n_windows=1,
        flag_shuffle=True,
        flag_replacement=True)

    ks_pred = ks_detector.predict(E_windows[0])
    mmd_pred = mmd_detector.predict(E_windows[0])
    lsdd_pred = lsdd_detector.predict(E_windows[0], return_p_val=True, return_distance=True)

    # print(ks_pred["data"]["is_drift"])
    ks_preds.append(ks_pred["data"]["is_drift"])
    mmd_preds.append(mmd_pred["data"]["is_drift"])
    lsdd_preds.append(lsdd_pred["data"]["is_drift"])





#ks_detector = KSDrift(E_train, p_val=.05)
#mmd_detector = MMDDrift(E_test, p_val=.05, n_permutations=100, backend="pytorch")
#lsdd_detector = LSDDDrift(E_test, backend='pytorch', p_val=.05)