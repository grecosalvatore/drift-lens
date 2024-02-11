# Run the Python script as a module
cd ..


python -m experiments.use_case_1_ag_news_science_drift.drift_detection_accuracy \
  --model_name 'distillbert' \
  --window_size 2000 \
  --drift_percentage 0.20 \
  --train_embedding_filename 'train_embedding_0_1_2.hdf5' \
  --test_embedding_file 'test_embedding_0_1_2.hdf5' \
  --new_unseen_embedding_file 'new_unseen_embedding_0_1_2.hdf5' \
  --drift_embedding_file 'drift_embedding_3.hdf5' \
  --output_dir 'ft_models_b/' \
  --learning_rate 5e-5 \
  --cuda \
  --verbose \
  --logging_steps 10 \
  --seed 42
