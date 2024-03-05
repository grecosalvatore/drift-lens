# Use case 1: Ag News with Science/Tech drifting label

This use case trains NLP models for topic detection with the Ag News dataset, and simulates a drift  by adding a new class label: Science/Tech. The original dataset has three class labels: World, Business, and Sport. The drift is simulated by adding a new class label: Science/Tech. The dataset is split into training and test sets. The models are trained on the training set and evaluated on the test set. The F1 score is used as the evaluation metric.
