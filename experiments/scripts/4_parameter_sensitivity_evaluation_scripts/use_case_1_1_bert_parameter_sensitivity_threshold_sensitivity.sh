# Run the Python script as a module
cd ../../..


python -m experiments.use_case_1_ag_news_science_drift.use_case_1_parameter_sensitivity_threshold_sensitivity \
  --number_of_runs 5 \
  --model_name 'bert' \
  --window_size 1000 \
  --number_of_windows 100 \
  --drift_percentage 0 5 10 15 20 \
  --threshold_sensitivity_list 0 1 5 10 25 \
  --batch_n_pc 150 \
  --per_label_n_pc 75 \
  --threshold_number_of_estimation_samples 10000 \
  --train_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/bert/train_embedding_0_1_2.hdf5' \
  --test_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/bert/test_embedding_0_1_2.hdf5' \
  --new_unseen_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/bert/new_unseen_embedding_0_1_2.hdf5' \
  --drift_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/bert/drift_embedding_3.hdf5' \
  --output_dir 'experiments/use_case_1_ag_news_science_drift/static/outputs/bert/' \
  --save_results \
  --cuda \
  --verbose \
  --seed 42
