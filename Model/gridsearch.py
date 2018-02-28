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
from h2o.frame import H2OFrame



def loadparse():
	
	###loading raw data
	#training

	training_data = h2o.import_file("/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/train_tagged.csv")

	print("\nTraining CSV file imported successfully")

	###Splitting training data into training, validation and testing 

	train, valid = training_data.split_frame([.7])

	print("Frame split correctly")

	#testing
	testing_data = h2o.import_file("/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/test_tagged.csv")

	print("\nTesting CSV file imported successfully")

	return train,valid



def predictresponse(train):

	###Defining predictor and response
	train["toxic"] = train["toxic"].asfactor()
	predictor = ["comment_text","tagged_length"]
	response = "toxic"
	print("Predictor and repsonese assigned")
	
	return train,predictor,response



def rungridsearch(hyperparams, folds, criteria, predictor, response, train, valid):

	###Define the grid search
	
	toxic_rdf_grid = H2OGridSearch(model=H2ORandomForestEstimator,
								  grid_id='toxic_rdf_grid_id',
								  hyper_params=hyperparams,
								  search_criteria=criteria)

	print("Grid search defined\n")

	###Running the grid search

	toxic_rdf_grid.train(predictor,
						response,
						training_frame=train,
						validation_frame=valid)#,
						 #nfolds = 5)
						 
	return toxic_rdf_grid



def logdata(toxic_rdf_grid, count, final_model_perf):
	                   
	###model performance sorted by logloss
	model_perf = pd.DataFrame(toxic_rdf_grid.sorted_metric_table()) #model perf as metric table of most recent gridsearch
	
	if count == 0:
		###model performance sorted by logloss
		final_model_perf = pd.DataFrame(toxic_rdf_grid.sorted_metric_table())  #initialises data frame
		print(final_model_perf)
		
		
	else:
		final_model_perf.append(model_perf, ignore_index=True) #appends new model data
		print(final_model_perf)
		
	return final_model_perf



def savedata(final_model_perf):
	
	written = False
	i = 0
	
	while written == False:
		
		if os.path.exists("/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Modelperformance/final_model_performance{0}.csv".format(i)) == True: ##Checks if file name already exists
			
			i = i + 1 
			
		else:
			
			final_model_perf.to_csv("/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Modelperformance/final_model_performance{0}.csv".format(i)) #Writes data to file
			print("Data written to file final_model_performance{0}.csv \n".format(i))
			break



def run (sample_rate_change, ntrees_change, count, final_model_perf):
	
	print("Model[{0}]".format(count))
	
	train, valid = loadparse() #sets train and valid using the returned variables from loadparse

	train, predictor, response = predictresponse(train) ##sets train predictor and valid using the returned variables predictrespsonse
	
	hyperparams, folds, criteria = changeparams(sample_rate_change, ntrees_change) ##sets hyperparams and number of folds for cross validation

	toxic_rdf_grid = rungridsearch(hyperparams, folds, criteria, predictor, response, train, valid) ##Runs gridsearch
	
	final_model_perf = logdata(toxic_rdf_grid, count, final_model_perf) #saves model
	
	h2o.remove_all()
	
	return final_model_perf 



def changeparams(sample_rate_change, ntrees_change):
	
	sample_rate_range = [0.1,0.2]#,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
	ntrees_range = [10,20]#,500,1000,2000,3000,4000]
	
	criteria = {'strategy': 'RandomDiscrete',
			   'max_models':1000,
			   'seed': 1234
			   }

	###Defining hyper params

	sample_rate_hp = sample_rate_range[sample_rate_change]
	ntrees_hp = ntrees_range[ntrees_change]
	folds = 5

	hyperparams= {"sample_rate": sample_rate_hp , "ntrees" : ntrees_hp}

	print("Hyper parameters and grid search parameters assigned")
	
	return hyperparams, folds, criteria



def iteration():
	
	count = 0
	final_model_perf = pd.DataFrame
	
	for sr_change in range(0,2):
		for nt_change in range(0,2):
			final_model_perf = run(sr_change, nt_change, count, final_model_perf)
			count = count + 1
			
	savedata(final_model_perf) #Writes model to file



iteration()


