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


def loadparse():

    # loading raw data
    # training

    training_data = h2o.import_file(
        "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/train_vectorised.csv")

    print("\nTraining CSV file imported successfully")

    # Splitting training data into training, validation and testing

    train, valid = training_data.split_frame([.7])

    print("Frame split correctly")

    # testing
    testing_data = h2o.import_file(
        "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/test_vectorised.csv")

    print("\nTesting CSV file imported successfully")

    return train, valid


def predictresponse(train):

    # Defining predictor and response
    train["toxic"] = train["toxic"].asfactor()
    train["comment_text"] = train["comment_text"].asfactor()
    predictor = "comment_text"
    response = "toxic"
    print("Predictor and response assigned")

    return train, predictor, response


def rungridsearch(
        hyperparams,
        folds,
        criteria,
        predictor,
        response,
        train,
        valid):

    # Define the grid search

    toxic_rdf_grid = H2OGridSearch(model=H2ORandomForestEstimator,
                                   grid_id='toxic_rdf_grid_id',
                                   hyper_params=hyperparams,
                                   search_criteria=criteria)

    print("Grid search defined\n")

    # Running the grid search

    toxic_rdf_grid.train(predictor,
                         response,
                         training_frame=train,
                         validation_frame=valid)  # ,
    # nfolds = 5)

    return toxic_rdf_grid


def logdata(toxic_rdf_grid, count, final_model_perf):

    # model performance sorted by logloss
    # model perf as metric table of most recent gridsearch
    model_perf = pd.DataFrame(toxic_rdf_grid.sorted_metric_table())

    if count == 0:
        # model performance sorted by logloss
        final_model_perf = pd.DataFrame(
            toxic_rdf_grid.sorted_metric_table())  # initialises data frame
        print(final_model_perf)
        print(count)

    else:

        final_model_perf = final_model_perf.append(
            model_perf, ignore_index=True)  # appends new model data
        print(final_model_perf)
        print(count)

    return final_model_perf


def savedata(final_model_perf):

    written = False
    i = 0

    while written == False:

        if os.path.exists("/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Modelperformance/final_model_performance{0}.csv".format(
                i)):  # Checks if file name already exists

            i = i + 1

        else:

            final_model_perf.to_csv(
                "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Modelperformance/final_model_performance{0}.csv".format(i))  # Writes data to file
            print(
                "Data written to file final_model_performance{0}.csv \n".format(i))
            break


def save_model(toxic_rdf_grid):

    models_in_grid = toxic_rdf_grid.get_grid(
        sort_by="logloss", decreasing=False)  # retrieves models from grid

    model = models_in_grid.models[0]

    i = 0
    saved = False

    while saved == False:

        if os.path.exists(
                "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Models/Model{0}".format(i)):

            i = i + 1

        else:

            h2o.save_model(
                model=model,
                path="/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Models/Model{0}".format(i),
                force=True)
            saved = True


def run(sample_rate_change, ntrees_change, count, final_model_perf):

    print("Model[{0}]".format(count))

    # sets train and valid using the returned variables from loadparse
    train, valid = loadparse()

    # sets train predictor and valid using the returned variables
    # predictrespsonse
    train, predictor, response = predictresponse(train)

    # sets hyperparams and number of folds for cross validation
    hyperparams, folds, criteria = changeparams(
        sample_rate_change, ntrees_change)

    toxic_rdf_grid = rungridsearch(
        hyperparams,
        folds,
        criteria,
        predictor,
        response,
        train,
        valid)  # Runs gridsearch

    final_model_perf = logdata(
        toxic_rdf_grid,
        count,
        final_model_perf)  # saves model information

    save_model(toxic_rdf_grid)  # saves binary model

    return final_model_perf


def changeparams(sample_rate_change, ntrees_change):

    sample_rate_range = [
        0.004,
        0.00425,
        0.0045,
        0.00475,
        0.005,
        0.00525,
        0.0055,
        0.00575,
        0.006,
        0.00625,
        0.0065,
        0.0675]
    ntrees_range = [
        1000,
        1250,
        1500,
        1750,
        2000,
        2250,
        2500,
        2750,
        3000,
        3250,
        3500,
        3750,
        4000,
        4250,
        4500,
        4750,
        5000]

    criteria = {'strategy': 'RandomDiscrete',
                'max_models': 10000,
                'seed': 1234
                }

    # Defining hyper params

    sample_rate_hp = sample_rate_range[sample_rate_change]
    ntrees_hp = ntrees_range[ntrees_change]
    folds = 5

    hyperparams = {"sample_rate": sample_rate_hp, "ntrees": ntrees_hp}

    print("Hyper parameters and grid search parameters assigned")

    return hyperparams, folds, criteria


def iteration():

    count = 0
    final_model_perf = pd.DataFrame

    for sr_change in range(0, 12):
        for nt_change in range(0, 17):
            final_model_perf = run(
                sr_change, nt_change, count, final_model_perf)
            h2o.remove_all()
            count = + 1

    savedata(final_model_perf)  # Writes model to file


iteration()
