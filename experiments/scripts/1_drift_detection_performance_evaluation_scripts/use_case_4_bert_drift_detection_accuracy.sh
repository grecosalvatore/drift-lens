# Run the Python script as a module
cd ../../..


python -m experiments.use_case_4_biasinbios_bert_female_drift.use_case_4_drift_detection_accuracy \
  --number_of_runs 5 \
  --model_name 'bert' \
  --window_size 500 \
  --number_of_windows 100 \
  --drift_percentage 0 5 10 15 20  \
  --threshold_sensitivity 1 \
  --threshold_number_of_estimation_samples 100 \
  --batch_n_p 150 \
  --per_label_n_pc 25 \
  --n_subsamples_mmd 8500 \
  --n_subsamples_lsdd 14000 \
  --n_subsamples_cvm -1 \
  --n_subsamples_ks -1 \
  --train_embedding_filepath 'experiments/use_case_4_biasinbios_bert_female_drift/static/saved_embeddings/bert/train_embedding.hdf5' \
  --test_embedding_filepath 'experiments/use_case_4_biasinbios_bert_female_drift/static/saved_embeddings/bert/test_embedding.hdf5' \
  --new_unseen_embedding_filepath 'experiments/use_case_4_biasinbios_bert_female_drift/static/saved_embeddings/bert/new_unseen_embedding.hdf5' \
  --drift_embedding_filepath 'experiments/use_case_4_biasinbios_bert_female_drift/static/saved_embeddings/bert/drift_embedding.hdf5' \
  --output_dir 'experiments/use_case_4_biasinbios_bert_female_drift/static/outputs/bert/' \
  --save_results \
  --cuda \
  --verbose \
  --seed 42 \
  --run_driftlens