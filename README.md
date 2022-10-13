# Graph Neural Networks Extract High-Resolution Cultivated Land Maps from Sentinel-2 Image Series
**Lukasz Tulczyjew, Michal Kawulok, Nicolas Longépé, Bertrand Le Saux and Jakub Nalepa**

(Submitted to IEEE GEOSCIENCE AND REMOTE SENSING LETTERS)

This repository contains the supplementary material to the above-mentioned paper. Specifically, it encompasses our implementation of the graph convolutional neural networks for extracting high-resolution cultivated land maps from the Sentinel-2 image series.
The code contains:
* ```gcnn.py``` Contains the implementation of the graph model.
* ```unet.py``` Holds the implementation of the U-Net model.
* ```runner.py``` Implements the main loop of the program and constitutes a starting point.
* ```random_forest.py``` Holds the implementation of the random forest method.
* ```deep_models_utils.py``` Contains the utilities for training both GCNNs and U-Nets.
* ```predict.py``` Contains the main script for running predictions for already trained models on all images (for visualization purposes).
* ```utils.py``` Holds the main utilities for the entire pipeline, including calculating metrics, logging, etc.
* ```predict.py``` Contains the main script for running inference on the simulated test data.
* ```visualize.py``` Implements the main script for saving RGB visualizations for specific images.
* ```dataset.py``` Implements the main logic for handling the dataset e.g., creating splits, loading, standardizing, etc.
