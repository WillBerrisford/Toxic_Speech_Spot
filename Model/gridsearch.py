###Importing Modules

import matplotlib.pyplot as plt
import numpy as np
import h2o
import pandas as pd
import csv
import os.path

###Removing current instances of H2o and initialising H2o

h2o.init(ip='localhost', nthreads=10,
					 min_mem_size='1G', max_mem_size='8G')
h2o.remove_all()

###Importing H2o

from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.model.metrics_base import H2OBinomialModelMetrics 
from h2o.grid.metrics import H2OBinomialGridSearch
from h2o.grid.grid_search import H2OGridSearch
from h2o.model import H2OBinomialModel
from h2o.model.model_base import ModelBase


def loadparse():
	
	###loading raw data
	#training

	training_data = h2o.import_file("C:\ComputerScience\DeepLearning\GitProjects\\toxicspeechspot\\toxicspeechspot\Data\\TaggedData\\train_tagged.csv")

	print("Training CSV file imported successfully")

	###Splitting training data into training, validation and testing 

	train, valid = training_data.split_frame([.7])

	print("Frame split correctly")

	#testing
	testing_data = h2o.import_file("C:\ComputerScience\DeepLearning\GitProjects\\toxicspeechspot\\toxicspeechspot\Data\\TaggedData\\test_tagged.csv")

	print("Testing CSV file imported successfully")

	return train,valid



def predictresponse(train):

	###Defining predictor and response
	train["toxic"] = train["toxic"].asfactor()
	predictor = ["comment_text","tagged_length"]
	response = "toxic"
	print("Predictor and repsonese assigned")
	
	return train,predictor,response
	
	

	
def  setparams():

	criteria = {'strategy': 'RandomDiscrete',
			   'max_models':1000,
			   'seed': 1234
			   }

	###Defining hyper params

	sample_rate_hp = 0.1#[i * 0.1 for i in range(1)]
	ntrees_hp = [100]
	folds = 5

	hyperparams= {"sample_rate": sample_rate_hp , "ntrees" : ntrees_hp}

	print("Hyper parameters and grid search parameters assigned")
	
	return hyperparams, folds, criteria





def rungridsearch(hyper_params, criteria, predictor, response, train, valid):

	###Define the grid search
	
	toxic_rdf_grid = H2OGridSearch(model=H2ORandomForestEstimator,
								  grid_id='toxic_rdf_grid_id',
								  hyper_params=hyperparams,
								  search_criteria=criteria)

	print("Grid search defined")

	###Running the grid search

	toxic_rdf_grid.train(predictor,
						response,
						training_frame=train,
						validation_frame=valid)#,
						 #nfolds = 5)
						 
	return toxic_rdf_grid
	



def savemodel(toxic_rdf_grid):
	                   
	###model performance sorted by logloss
	model_perf = toxic_rdf_grid.sorted_metric_table() 

	written = False
	i = 0

	while written == False:
		if os.path.exists("C:\ComputerScience\DeepLearning\GitProjects\\toxicspeechspot\\toxicspeechspot\ModelPerformance\model_performance{0}.csv".format(i)) == True:
			i = i + 1
		else:
			model_perf.to_csv("C:\ComputerScience\DeepLearning\GitProjects\\toxicspeechspot\\toxicspeechspot\ModelPerformance\model_performance{0}.csv".format(i))
			break




train, valid = loadparse() #sets train and valid using the returned variables from loadparse


train, predictor, response = predictresponse(train) ##sets train predictor and valid using the returned variables predictrespsonse


hyperparams, folds, criteria = setparams() ##sets hyperparams and number of folds for cross validation

toxic_rdf_grid = rungridsearch(hyperparams, criteria, predictor, response, train, valid)

savemodel(toxic_rdf_grid)










