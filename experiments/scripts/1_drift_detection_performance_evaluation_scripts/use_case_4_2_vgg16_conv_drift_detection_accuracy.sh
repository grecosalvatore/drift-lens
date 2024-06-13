# Run the Python script as a module
cd ../../..


python -m experiments.use_case_4_intel_image_sea_drift.use_case_4_drift_conv_detection_accuracy \
  --number_of_runs 5 \
  --model_name 'vgg16' \
  --window_size 1000 \
  --number_of_windows 100 \
  --drift_percentage 0 5 10 15 20 \
  --threshold_sensitivity 99 \
  --threshold_number_of_estimation_samples 10000 \
  --batch_n_p 150 \
  --per_label_n_pc 25 \
  --n_subsamples_mmd 8500 \
  --n_subsamples_lsdd 14000 \
  --n_subsamples_cvm -1 \
  --n_subsamples_ks -1 \
  --train_embedding_filepath 'experiments/use_case_4_intel_image_sea_drift/static/saved_embeddings/vgg16/train_embedding_conv.hdf5' \
  --test_embedding_filepath 'experiments/use_case_4_intel_image_sea_drift/static/saved_embeddings/vgg16/test_embedding_conv.hdf5' \
  --new_unseen_embedding_filepath 'experiments/use_case_4_intel_image_sea_drift/static/saved_embeddings/vgg16/new_unseen_embedding_conv.hdf5' \
  --drift_embedding_filepath 'experiments/use_case_4_intel_image_sea_drift/static/saved_embeddings/vgg16/drift_embedding_conv.hdf5' \
  --output_dir 'experiments/use_case_4_intel_image_sea_drift/static/outputs/vgg16/' \
  --save_results \
  --cuda \
  --verbose \
  --seed 42
