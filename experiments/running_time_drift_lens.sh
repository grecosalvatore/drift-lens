# Run the Python script as a module
cd ..


python -m experiments.running_time_comparison \
  --number_of_runs 5 \
  --training_label_list 0 1 2 \
  --drift_label_list 3 \
  --run_driftlens \
  --run_experiment_reference_window \
  --run_experiment_window_size \
  --reference_window_size_range '50000:500000:50000' \
  --datastream_window_size_range '1000:10000:1000' \
  --embedding_dimensionality_range '500:2500:500' \
  --fixed_reference_window_size 50000 \
  --fixed_datastream_window_size 5000 \
  --fixed_embedding_dimensionality 1000 \
  --batch_n_p 150 \
  --per_label_n_pc 75 \
  --output_dir 'experiments/' \
  --save_results \
  --verbose \
  --seed 42
