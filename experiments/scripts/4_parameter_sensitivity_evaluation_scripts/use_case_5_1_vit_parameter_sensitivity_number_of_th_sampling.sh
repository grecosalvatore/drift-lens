# Run the Python script as a module
cd ../../..


python -m experiments.use_case_6_stl_truck_drift.use_case_6_parameter_sensitivity_number_of_th_sampling \
  --number_of_runs 5 \
  --model_name 'vit' \
  --window_size 1000 \
  --number_of_windows 100 \
  --drift_percentage 0 5 10 15 20 \
  --threshold_sensitivity 99 \
  --threshold_number_of_estimation_samples_list 100 1000 5000 10000 25000 \
  --batch_n_p 150 \
  --per_label_n_pc 25 \
  --train_embedding_filepath 'experiments/use_case_6_stl_truck_drift/static/saved_embeddings/vit/train_embedding.hdf5' \
  --test_embedding_filepath 'experiments/use_case_6_stl_truck_drift/static/saved_embeddings/vit/test_embedding.hdf5' \
  --new_unseen_embedding_filepath 'experiments/use_case_6_stl_truck_drift/static/saved_embeddings/vit/new_unseen_embedding.hdf5' \
  --drift_embedding_filepath 'experiments/use_case_6_stl_truck_drift/static/saved_embeddings/vit/drift_embedding.hdf5' \
  --output_dir 'experiments/use_case_6_stl_truck_drift/static/outputs/vit/' \
  --save_results \
  --cuda \
  --verbose \
  --seed 42
