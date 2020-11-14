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

## Authors

### Original TMC Shapely and LOOV code
* **Amirata Ghorbani** - [Website](http://web.stanford.edu/~amiratag)
* **James Zou** - [Website](https://www.james-zou.com/)
### Experiments and alterations to original code
* **Josh Mannion**
* **Lawton Manning**
