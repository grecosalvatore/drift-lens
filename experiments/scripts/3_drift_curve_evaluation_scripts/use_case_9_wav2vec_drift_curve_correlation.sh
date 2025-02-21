# Run the Python script as a module
cd ../../..


python -m experiments.drift_curve_correlation \
  --training_label_list 0 1  \
  --drift_label_list 2 \
  --number_of_runs 10 \
  --model_name 'wav2vec' \
  --window_size 1000 \
  --number_of_windows 100 \
  --sudden_drift_offset 50 \
  --sudden_drift_percentage 0.25 \
  --incremental_starting_drift_percentage 0.1 \
  --incremental_drift_increase_rate 0.01 \
  --incremental_drift_offset 20 \
  --periodic_drift_percentage 0.4 \
  --periodic_drift_offset 20 \
  --periodic_drift_duration 20 \
  --batch_n_p 25 \
  --per_label_n_pc 25 \
  --train_embedding_filepath 'experiments/use_case_9_common_voice_gender_classification_accent_drift/static/saved_embeddings/wav2vec/train_embedding.hdf5' \
  --test_embedding_filepath 'experiments/use_case_9_common_voice_gender_classification_accent_drift/static/saved_embeddings/wav2vec/test_embedding.hdf5' \
  --new_unseen_embedding_filepath 'experiments/use_case_9_common_voice_gender_classification_accent_drift/static/saved_embeddings/wav2vec/new_unseen_embedding.hdf5' \
  --drift_embedding_filepath 'experiments/use_case_9_common_voice_gender_classification_accent_drift/static/saved_embeddings/wav2vec/drift_embedding.hdf5' \
  --output_dir 'experiments/use_case_9_common_voice_gender_classification_accent_drift/static/outputs/wav2vec/' \
  --save_results \
  --verbose \
  --seed 42
