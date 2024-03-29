# Run the Python script as a module
cd ../../..


python -m experiments.use_case_1_ag_news_science_drift.use_case_1_drift_detection_accuracy \
  --number_of_runs 5 \
  --model_name 'roberta' \
  --window_size 2000 \
  --number_of_windows 10 \
  --drift_percentage 0 5 10 \
  --threshold_sensitivity 99 \
  --threshold_number_of_estimation_samples 10 \
  --batch_n_p 150 \
  --per_label_n_pc 75 \
  --n_subsamples_sota 500 \
  --train_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/roberta/train_embedding_0_1_2.hdf5' \
  --test_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/roberta/test_embedding_0_1_2.hdf5' \
  --new_unseen_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/roberta/new_unseen_embedding_0_1_2.hdf5' \
  --drift_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/roberta/drift_embedding_3.hdf5' \
  --output_dir 'experiments/use_case_1_ag_news_science_drift/static/outputs/roberta/' \
  --save_results \
  --cuda \
  --verbose \
  --seed 42
