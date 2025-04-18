# Run the Python script as a module
cd ../../..


python -m experiments.running_time_comparison_reference_window_size \
  --number_of_runs 5 \
  --training_label_list 0 1 2 \
  --drift_label_list 3 \
  --run_ks \
  --run_cvm \
  --run_experiment_reference_window \
  --reference_window_size_range '50000:70000:10000' \
  --fixed_reference_window_size 5000 \
  --fixed_datastream_window_size 1000 \
  --fixed_embedding_dimensionality 1000 \
  --batch_n_p 150 \
  --per_label_n_pc 75 \
  --output_dir 'experiments/' \
  --save_results \
  --verbose \
  --seed 42
