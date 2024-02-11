# Run the Python script as a module
cd ..


python -m experiments.use_case_1_ag_news_science_drift.drift_detection_accuracy \
  --model_name 'roberta' \
  --window_size 2000 \
  --drift_percentage 0.20 \
  --train_embedding_file '/home/mlaquatra/nlp_projects/geolingit/geolingit-data/standard-track/subtask_b/train_b.tsv' \
  --test_embedding_file '/home/mlaquatra/nlp_projects/geolingit/geolingit-data/standard-track/subtask_b/dev_b.tsv' \
  --new_unseen_embedding_file '/home/mlaquatra/nlp_projects/geolingit/geolingit-data/standard-track/subtask_b/dev_b.tsv' \
  --drift_embedding_file '/home/mlaquatra/nlp_projects/geolingit/geolingit-data/standard-track/subtask_b/dev_b.tsv' \
  --output_dir 'ft_models_b/' \
  --learning_rate 5e-5 \
  --cuda \
  --verbose \
  --logging_steps 10 \
  --seed 42
