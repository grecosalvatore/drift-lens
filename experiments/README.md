# Experiments reproducibility
This folder contains the scripts and instructions to reproduce the experiments of the paper.

## Table of Contents
- [Experimental use cases](#experimental-use-cases)
- [Drift detection performance evaluation](#drift-detection-performance-evaluation)
- [Complexity evaluation](#complexity-evaluation)
- [Drift curve evaluation](#drift-curve-evaluation)
- [Parameter sensitivity evaluation](#parameter-sensitivity-evaluation)



## Experimental use cases


Each macro use case is divided into folder. 
For each macro use case, there is a folder `model_training_and_embedding_extraction` containing the following scripts:
- `{model}_training.ipynb`: Jupyter notebook to train the model
- `{model}_embedding_extraction.py`: Jupyter notebook to extract the embedding representations the model

Each macro use case also contains a README.md file with specific information about the use case and the simulated drift.

<table>
  <caption>Experimental use cases.</caption>
  <thead>
    <tr>
      <th rowspan="3">Data Type</th>
      <th rowspan="3">Dataset</th>
        <th>Use Case</th>
      <th>Models</th>
      <th>F1</th>
      <th rowspan="3">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3">Text</th>
      <th rowspan="3">Ag News</th>
      <th>1.1</th>
      <td>BERT</td>
      <td>0.98</td>
      <td rowspan="3"> <b>Task</b>: Topic Classification. <BR>
<b>Training Labels</b>: <i>World</i>, <i>Business</i>, and <i>Sport</i> <BR>
<b>Drift</b>: Simulated with one new class label: <i>Science/Tech</i></td>
    </tr>
    <tr>
      <th>1.2</th>
      <td>DistillBERT</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>1.3</th>
      <td>RoBERTa</td>
      <td>0.98</td>
    </tr>
    <tr class="separator">
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  <tr>
      <th rowspan="3">Text</th>
      <th rowspan="3">20 Newsgroup</th>
      <th>2.1</th>
      <td>BERT</td>
      <td>0.88</td>
      <td rowspan="3"><b>Task</b>: Topic Classification. <BR>
<b>Training Labels</b>: <i>Technology</i>, <i>Sale-Ads</i>, <i>Politics</i>, <i>Religion</i>, <i>Science</i> <BR>
<b>Drift</b>: Simulated with one new class label: <i>Recreation</i></td>
    </tr>
    <tr>
      <th>2.2</th>
      <td>DistillBERT</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>2.3</th>
      <td>RoBERTa</td>
      <td>0.88</td>
    </tr>
    <tr class="separator">
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="3">Text</th>
      <th rowspan="3">20 Newsgroup</th>
      <th rowspan="3">3</th>
      <td rowspan="3">BERT</td>
      <td rowspan="3">0.87</td>
      <td rowspan="3"><b>Task</b>: Topic Classification. <BR>
    <b>Training Labels</b>: <i>Technology</i>, <i>Sale-Ads</i>, <i>Politics</i>, <i>Religion</i>, <i>Science</i> <BR>
    <b>Drift</b>: Simulated with one new class label: <i>Recreation</i></td>
    </tr>
    <tr>
    </tr>
    <tr>
    </tr>
    <tr class="separator">
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  <tr>
      <th rowspan=2">Image</th>
      <th rowspan="2">Intel-Image</th>
      <th>4.1</th>
      <td>VGG16</td>
      <td>0.89</td>
      <td rowspan="2"> <b>Task</b>: Image Classification. <BR>
<b>Training Labels</b>: <i>Forest</i>, <i>Glacier</i>, <i>Mountain</i>, <i>Building</i>, <i>Street</i> <BR>
<b>Drift</b>: Simulated with one new class label: <i>Sea</i></td>
    </tr>
    <tr>
      <th>4.2</th>
      <td>VisionTransformer</td>
      <td>0.90</td>
    </tr>
    <tr class="separator">
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  <tr>
      <th rowspan="2">Image</th>
      <th rowspan="2">STL-10</th>
      <th>5.1</th>
      <td>VGG16</td>
      <td>0.82</td>
      <td rowspan="2"> <b>Task</b>: Image Classification. <BR>
<b>Training Labels</b>: <i>Airplane</i>, <i>Bird</i>, <i>Car</i>, <i>Cat</i>, <i>Deer</i>, <i>Dog</i>, <i>Horse</i>, <i>Monkey</i>, <i>Ship</i> <BR>
<b>Drift</b>: Simulated with one new class label: <i>Truck</i></td>
    </tr>
    <tr>
      <th>5.2</th>
      <td>VisionTransformer</td>
      <td>0.96</td>
    </tr>
    <tr class="separator">
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="3">Image</th>
      <th rowspan="3">STL-10</th>
      <th rowspan="3">6</th>
      <td rowspan="3">VisualTransformer</td>
      <td rowspan="3">0.90</td>
      <td rowspan="3"><b>Task</b>: Image Classification. <BR>
    <b>Training Labels</b>: <i>Airplane</i>, <i>Bird</i>, <i>Car</i>, <i>Cat</i>, <i>Deer</i>, <i>Dog</i>, <i>Horse</i>, <i>Monkey</i>, <i>Ship</i>, <i>Truck</i> <BR>
 <b>Drift</b>: Simulated with one new class label: <i>Recreation</i></td>
    </tr>
    <tr>
    </tr>
    <tr>
    </tr>
    <tr class="separator">
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
<tr>
      <th rowspan="3">Speech</th>
      <th rowspan="3">Common Voice</th>
      <th rowspan="3">7</th>
      <td rowspan="3">Wav2Vec</td>
      <td rowspan="3">0.91</td>
      <td rowspan="3"><b>Task</b>: Gender Classification. <BR>
    <b>Training Labels</b>: <i>Male</i>, <i>Female</i> (English - US and UK accents) <BR>
    <b>Drift</b>: Simulated with speeches from difference accents (English - Australian, Canadian, Scottish)</td>
    </tr>
    <tr>
    </tr>
    <tr>
    </tr>
    <tr class="separator">
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

## Drift detection performance evaluation
This evaluation aims to determine the effectiveness of DriftLens in detecting windows containing drifted samples of varying severity.
The drift detection problem is tackled as a binary classification task. The task consists in predicting whether a window of new samples contains drift. 

The `scripts/1_drift_detection_performance_evaluation_scripts` folder contains the scripts to run the evaluation for all the use cases. These scripts can be used to reproduce Tables 3 and 4 in the paper.

An example of drift detection performance evaluation script is the following:

```bash
cd ../../..


python -m experiments.use_case_1_ag_news_science_drift.use_case_1_drift_detection_accuracy \
  --number_of_runs 10 \
  --model_name 'bert' \
  --window_size 2000 \
  --number_of_windows 100 \
  --drift_percentage 0 5 10 15 20 \
  --threshold_sensitivity 99 \
  --threshold_number_of_estimation_samples 10000 \
  --batch_n_p 150 \
  --per_label_n_pc 75 \
  --n_subsamples_sota 5000 \
  --train_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/bert/train_embedding_0_1_2.hdf5' \
  --test_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/bert/test_embedding_0_1_2.hdf5' \
  --new_unseen_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/bert/new_unseen_embedding_0_1_2.hdf5' \
  --drift_embedding_filepath 'experiments/use_case_1_ag_news_science_drift/static/saved_embeddings/bert/drift_embedding_3.hdf5' \
  --output_dir 'experiments/use_case_1_ag_news_science_drift/static/outputs/bert/' \
  --save_results \
  --cuda \
  --verbose \
  --seed 42
```

## Complexity evaluation
This evaluation aims to ascertain the effectiveness of DriftLens to
perform near real-time drift detection. To this end, we compare the
running time of the drift detectors by varying the reference and
data stream windows sizes, and the embedding dimensionality.

The folder  contains the script to run the evaluation. These scripts can be used to reproduce Figures 5 and 6 in the paper.
The script is organized as follows:

## Drift curve evaluation
This evaluation aims to measure the ability of DriftLens to coherently represent and characterize the drift curve.

The folder  contains the script to run the evaluation. These scripts can be used to reproduce Table 5 in the paper.
The script is organized as follows:

## Parameter sensitivity evaluation
This evaluation aims to determine the robustness and sensitivity of DriftLens to its parameters.

The folder  contains the script to run the evaluation. These scripts can be used to reproduce Table 6 in the paper.
The script is organized as follows: