#
<div align="center">
  <img src="https://github.com/grecosalvatore/drift-lens/raw/main/docs/_static/images/Drift_Lens_Logo.png" width="300"/>
  <h3>Unsupervised Concept Drift Detection from Deep Learning
Representations in Real-time</h3>
</div>
<br/>

[![Documentation Status](https://readthedocs.org/projects/driftlens/badge/?version=latest)](https://driftlens.readthedocs.io/en/latest/?version=latest)
[![Version](https://img.shields.io/pypi/v/driftlens?color=blue)](https://pypi.org/project/driftlens)
[![License](https://img.shields.io/github/license/grecosalvatore/drift-lens)](https://github.com/grecosalvatore/drift-lens/blob/main/LICENSE)
[![arxiv preprint](https://img.shields.io/badge/arXiv-2406.17813-b31b1b.svg)](https://arxiv.org/abs/2406.17813)
[![Downloads](https://static.pepy.tech/badge/driftlens)](https://pepy.tech/project/driftlens)

*DriftLens* is an **unsupervised** framework for real-time *concept drift* **detection** and **characterization**. 
It is designed for deep learning classifiers handling unstructured data, and leverages distribution distances in deep learning representations to enable efficient and accurate detection.


## Publications
The latest advancements in the *DriftLens* methodology and its evaluation has been published in the paper: \
[Unsupervised Concept Drift Detection from Deep Learning Representations in Real-time](https://ieeexplore.ieee.org/document/11103500) **(Greco et al., 2025)**, *IEEE Transactions on Knowledge and Data Engineering (TKDE).*


The preliminary idea was first proposed in the paper: \
[Drift Lens: Real-time unsupervised Concept Drift detection by evaluating per-label embedding distributions](https://ieeexplore.ieee.org/document/9679880) **(Greco et al., 2021)**, *nternational Conference on Data Mining Workshops (ICDMW).*

*DriftLens* as been also implemented in a Flask application tool ([GitHub](https://github.com/grecosalvatore/DriftLensDemo)): \
[DriftLens: A Concept Drift Detection Tool](https://openproceedings.org/2024/conf/edbt/paper-239.pdf) **(Greco et al., 2024)**, *International Conference on Extending Database Technology (EDBT) Demo*.

## Table of Contents
- [Installation](#installation)
- [Example of usage](#example-of-usage)
- [DriftLens Methodology](#driftlens-methodology)
- [Experiments Reproducibility](#experiments-reproducibility)
- [References](#references)
- [Authors](#authors)

## Installation
DriftLens is available on PyPI and can be installed with pip for Python >= 3.
```bash
# Install latest stable version
pip install driftlens

# Alternatively, install latest development version
pip install git+https://github.com/grecosalvatore/drift-lens
```

## Example of usage
```python
from driftlens.driftlens import DriftLens

# DriftLens parameters
batch_n_pc = 150 # Number of principal components to reduce per-batch embeddings
per_label_n_pc = 75 # Number of principal components to reduce per-label embeddings
window_size = 1000 # Window size for drift detection
threshold_number_of_estimation_samples = 10000 # Number of sampled windows to estimate the threshold values

# Initialize DriftLens
dl = DriftLens()

# Estimate the baseline (offline phase)
baseline = dl.estimate_baseline(E=E_train,
                                Y=Y_predicted_train,
                                label_list=training_label_list,
                                batch_n_pc=batch_n_pc,
                                per_label_n_pc=per_label_n_pc)

# Estimate the threshold values with DriftLens (offline phase)
per_batch_distances_sorted, per_label_distances_sorted = dl.random_sampling_threshold_estimation(
                                                            label_list=training_label_list,
                                                            E=E_test,
                                                            Y=Y_predicted_test,
                                                            batch_n_pc=batch_n_pc,
                                                            per_label_n_pc=per_label_n_pc,
                                                            window_size=window_size,
                                                            n_samples=threshold_number_of_estimation_samples,
                                                            flag_shuffle=True,
                                                            flag_replacement=True)

# Compute the window distribution distances (Frechet Inception Distance) with DriftLens
dl_distance = dl.compute_window_distribution_distances(E_windows[0], Y_predicted_windows[0])

```

## DriftLens Methodology
<div align="center">
  <img src="docs/_static/images/drift-lens-architecture.png" width="600"/>
  <h4>DriftLens Methodology.</h4>
</div>

The methodology includes an *offline* and an *online* phases. 


In the *offline* phase, DriftLens, takes in input a historical dataset (i.e., baseline and threshold datasets), then: 

(1) Estimates the reference distributions from the baseline dataset (e.g., training dataset). The reference
distributions, called **baseline**, represent the distribution of features (i.e., embedding) that the model has learned during the training phase (i.e., they represent the absence of drift).
(2) Estimates threshold distance values from the threshold dataset to discriminate between drift and no-drift conditions.

In the *online* phase, the new data stream is processed in windows of fixed size. For each window, DriftLens:

(3) Estimates the distributions of the new data windows 
(4) it computes the distribution distances with respect to the reference distributions
(5) it evaluates the distances against the threshold values.  If the distance exceeds the threshold, the presence of drift is predicted.

In both phases, the distributions are estimated as multivariate normal distribution by computing the mean and the covariance over the embedding vectors.

DriftLens uses the Frechet Distance to measure the similarity between the reference (i.e., baseline) and the new window distributions.

## Experiments Reproducibility
Instructions and scripts for the experimental evaluation reproducibility are located in the [experiments folder](experiments/README.md).

## Future Developments
- ⚙️ Drift explanations
- 📊 DriftLens visualization

## References
If you use DriftLens, please cite the following papers:

1) DriftLens methodology and evaluation has been accepted at the IEEE Transactions on Knowledge and Data Engineering (TKDE):
```bibtex
@ARTICLE{11103500,
  author={Greco, Salvatore and Vacchetti, Bartolomeo and Apiletti, Daniele and Cerquitelli, Tania},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Unsupervised Concept Drift Detection from Deep Learning Representations in Real-time}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  keywords={Concept drift;Deep learning;Data models;Computational modeling;Real-time systems;Adaptation models;Detectors;Complexity theory;Production;Monitoring;Concept Drift;Data Drift;Drift Detection;Drift Explanation;Deep Learning;NLP;Computer Vision;Audio},
  doi={10.1109/TKDE.2025.3593123}
}
```

2) Preliminary idea 
```bibtex
@INPROCEEDINGS{driftlens,
  author={Greco, Salvatore and Cerquitelli, Tania},
  booktitle={2021 International Conference on Data Mining Workshops (ICDMW)}, 
  title={Drift Lens: Real-time unsupervised Concept Drift detection by evaluating per-label embedding distributions}, 
  year={2021},
  volume={},
  number={},
  pages={341-349},
  doi={10.1109/ICDMW53433.2021.00049}
  }
```

3) Webapp tool
```bibtex
@inproceedings{greco2024driftlens,
  title={DriftLens: A Concept Drift Detection Tool},
  author={Greco, Salvatore and Vacchetti, Bartolomeo and Apiletti, Daniele and Cerquitelli, Tania and others},
  booktitle={Advances in Database Technology},
  volume={27},
  pages={806--809},
  year={2024},
  organization={Open proceedings}
}
```

## Authors

- **Salvatore Greco**, *Politecnico di Torino* - [Homepage](https://grecosalvatore.github.io/) - [GitHub](https://github.com/grecosalvatore) - [Twitter](https://twitter.com/_salvatoregreco)
- **Bartolomeo Vacchetti**, *Politecnico di Torino* 
- **Daniele Apiletti**, *Politecnico di Torino* - [Homepage](https://www.polito.it/en/staff?p=daniele.apiletti)
- **Tania Cerquitelli**, *Politecnico di Torino* - [Homepage](https://dbdmg.polito.it/dbdmg_web/people/tania-cerquitelli/)