# Run the Python script as a module
cd ../../..


python -m experiments.use_case_7_common_voice_gender_classification_accent_drift.use_case_7_drift_detection_accuracy \
  --number_of_runs 5 \
  --model_name 'wav2vec' \
  --window_size 500 \
  --number_of_windows 100 \
  --drift_percentage 0 5 10 15 20 \
  --threshold_sensitivity 99 \
  --threshold_number_of_estimation_samples 10000 \
  --batch_n_pc 15 \
  --per_label_n_pc 25 \
  --n_subsamples_sota 5000 \
  --train_embedding_filepath 'experiments/use_case_7_common_voice_gender_classification_accent_drift/static/saved_embeddings/wav2vec/train_embedding.hdf5' \
  --test_embedding_filepath 'experiments/use_case_7_common_voice_gender_classification_accent_drift/static/saved_embeddings/wav2vec/test_embedding.hdf5' \
  --new_unseen_embedding_filepath 'experiments/use_case_7_common_voice_gender_classification_accent_drift/static/saved_embeddings/wav2vec/new_unseen_embedding.hdf5' \
  --drift_embedding_filepath 'experiments/use_case_7_common_voice_gender_classification_accent_drift/static/saved_embeddings/wav2vec/drift_embedding.hdf5' \
  --output_dir 'experiments/use_case_7_common_voice_gender_classification_accent_drift/static/outputs/wav2vec/' \
  --save_results \
  --cuda \
  --verbose \
  --seed 42
