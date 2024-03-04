# Run the Python script as a module
cd ../../..


python -m experiments.drift_curve_correlation \
  --training_label_list 0 1 2 \
  --drift_label_list 3 \
  --number_of_runs 10 \
  --model_name 'distillbert' \
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
  --batch_n_p 150 \
  --per_label_n_pc 75 \
  --train_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/distillbert/train_embedding_0_1_2.hdf5' \
  --test_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/distillbert/test_embedding_0_1_2.hdf5' \
  --new_unseen_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/distillbert/new_unseen_embedding_0_1_2.hdf5' \
  --drift_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/distillbert/drift_embedding_3.hdf5' \
  --output_dir 'experiments/use_case_1_ag_news_science_drift/static/outputs/distillbert/' \
  --save_results \
  --verbose \
  --seed 42
