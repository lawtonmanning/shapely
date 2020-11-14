Assignment 5
=====================================

This code is divided between two data sets: brunello and cycling.

## Prerequisites

- Python, NumPy, Tensorflow 2.3 Scikit-learn, Matplotlib

## Basic Usage

### Cycling
1. Run precycling.py (possibly changing data_dir to point to the train and test folders for cycling data)
This script will preprocess the cycling data and partition it according to the description in the report
2. Run cycling_shapely.py (again changing data_dir)
This script will load the data sets created from precycling.py and run the TMC Shapely and LOOV calculations using the models described in the reports.
It will also create histograms and other figures used in the report as well as run the Experiment 4 by grouping data by athlete.

### Brunello
1. brunello_shapley.py 
This script takes in the brunello data and perform the TMC Shapely and LOOV calculations (hardcoded for logistic and accuracy but was changed to produce figures in the report).
2. experiment5_brunello.py
This script takes in the brunello train data with the 100 TMC estimated shapely values and the brunello test data with 5000 total samples. A RF model is then trained on the 100 samples with shapley values to predict shapley values for 4000 of the 5000 samples. The remaining 1000 samples are used to evaluate model performance using accuracy while adding the most and least valuable points(determined by shapley) to the training data (og 100) sequentially and training and evaluating a gradient boosting classifier at each step. 
## Authors

### Original TMC Shapely and LOOV code
* **Amirata Ghorbani** - [Website](http://web.stanford.edu/~amiratag)
* **James Zou** - [Website](https://www.james-zou.com/)
### Experiments and alterations to original code
* **Josh Mannion**
* **Lawton Manning**
