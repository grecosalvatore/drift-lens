# Use case 1: Ag News with Science/Tech drifting label

This use case trains NLP models for topic detection with the Ag News dataset, and simulates a drift  by adding a new class label: *Science/Tech*. 

The original dataset has three class labels: *World*, *Business*, and *Sport*. <br>
Drift is simulated by adding a new class label: *Science/Tech*. 

The dataset is split into the following sets:
- **Training set**: Dataset used to fine-tune the classifiers (BERT, DistilBERT, RoBERTa). It comprises the classes *World*, *Business*, and *Sport*.
- **Test set**: Dataset used to evaluate the fine-tuned classifiers (BERT, DistilBERT, RoBERTa). It comprises the classes *World*, *Business*, and *Sport*.
- **New unseen**: Dataset used to generate the datastream. It 
- **Drifted**:

training and test sets. The models are trained on the training set and evaluated on the test set. The F1 score is used as the evaluation metric.
