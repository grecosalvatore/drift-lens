# Run the Python script as a module
cd ..


python -m experiments.running_time_comparison \
  --number_of_runs 3 \
  --training_label_list 0 1 2 \
  --drift_label_list 3 \
  --reference_window_size_range '500:1500:500' \
  --datastream_window_size_range '500:5000:500' \
  --fixed_reference_window_size 1000 \
  --fixed_datastream_window_size 1000 \
  --fixed_embedding_dimensionality 1000 \
  --batch_n_p 150 \
  --per_label_n_pc 75 \
  --model_name 'bert' \
  --train_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/bert/train_embedding_0_1_2.hdf5' \
  --test_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/bert/test_embedding_0_1_2.hdf5' \
  --new_unseen_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/bert/new_unseen_embedding_0_1_2.hdf5' \
  --drift_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/bert/drift_embedding_3.hdf5' \
  --output_dir 'experiments/' \
  --save_results \
  --verbose \
  --seed 42
