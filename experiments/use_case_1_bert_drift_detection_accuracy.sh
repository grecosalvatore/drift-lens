# Run the Python script as a module
cd ..


python -m experiments.use_case_1_ag_news_science_drift.drift_detection_accuracy \
  --model_name 'bert' \
  --window_size 2000 \
  --number_of_windows 10 \
  --drift_percentage 20 \
  --threshold_sensitivity 99 \
  --batch_n_p 150 \
  --per_label_n_pc 75 \
  --train_embedding_filename 'train_embedding_0_1_2.hdf5' \
  --test_embedding_filename 'test_embedding_0_1_2.hdf5' \
  --new_unseen_embedding_filename 'new_unseen_embedding_0_1_2.hdf5' \
  --drift_embedding_filename 'drift_embedding_3.hdf5' \
  --output_dir 'ft_models_b/' \
  --cuda \
  --verbose \
  --seed 42
