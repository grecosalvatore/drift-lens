# Run the Python script as a module
cd ../../..



python -m experiments.use_case_2_20_news_recreation_drift.use_case_2_drift_detection_accuracy_balanced \
  --number_of_runs 10 \
  --model_name 'bert' \
  --window_size 1000 \
  --number_of_windows 100 \
  --drift_percentage 0 5 10 15 20 \
  --threshold_sensitivity 99 \
  --threshold_number_of_estimation_samples 10000 \
  --batch_n_p 150 \
  --per_label_n_pc 75 \
  --n_subsamples_sota 5000 \
  --train_embedding_filepath 'experiments/use_case_2_20_news_recreation_drift/static/saved_embeddings/bert/train_embedding_0-4.hdf5' \
  --test_embedding_filepath 'experiments/use_case_2_20_news_recreation_drift/static/saved_embeddings/bert/test_embedding_0-4.hdf5' \
  --new_unseen_embedding_filepath 'experiments/use_case_2_20_news_recreation_drift/static/saved_embeddings/bert/new_unseen_embedding_0-4.hdf5' \
  --drift_embedding_filepath 'experiments/use_case_2_20_news_recreation_drift/static/saved_embeddings/bert/drift_embedding_5.hdf5' \
  --output_dir 'experiments/use_case_2_20_news_recreation_drift/static/outputs/bert/' \
  --save_results \
  --cuda \
  --verbose \
  --seed 42
  #--sota_comparison \
