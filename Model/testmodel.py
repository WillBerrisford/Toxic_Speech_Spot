# Importing Modules

import matplotlib.pyplot as plt
import numpy as np
import h2o
import pandas as pd
import csv
import os.path

# Removing current instances of H2o and initialising H2o

h2o.init(ip='localhost', nthreads=10,
         min_mem_size='1G', max_mem_size='7G')
h2o.remove_all()

# Importing H2o

from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.model.metrics_base import H2OBinomialModelMetrics
from h2o.grid.metrics import H2OBinomialGridSearch
from h2o.grid.grid_search import H2OGridSearch
from h2o.model import H2OBinomialModel
from h2o.model.model_base import ModelBase
from h2o.frame import H2OFrame


# loading raw datapytho
# training

training_data = h2o.import_file(
    "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/train_tagged.csv")

print("\nTraining CSV file imported successfully")

# Splitting training data into training, validation and testing

train, valid = training_data.split_frame([.7])

print("Frame split correctly")

# testing
testing_data = h2o.import_file(
    "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/test_tagged.csv")

print("\nTesting CSV file imported successfully")


# Defining predictor and response
train["toxic"] = train["toxic"].asfactor()
predictor = ["comment_text", "tagged_length"]
response = "toxic"
print("Predictor and repsonese assigned")


# Define the grid search

toxic_rdf_grid = h2o.load_model(
    "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Notable_Models/Model427/toxic_rdf_grid_id_model_0")

print("Grid search defined\n")

# Running the grid search


toxic_rdf_grid.train(predictor,
                     response,
                     training_frame=train,
                     validation_frame=valid)  # ,
# nfolds = 5)


print(toxic_rdf_grid.confusion_matrix)
print(toxic_rdf_grid.confusion_matrix)
