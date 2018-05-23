#import modules
import pandas as pd
import numpy as np

training_data = pd.read_csv(
        "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/train_vectorised.csv")
training_name = ("train_vectorised_shrunk")

training_data_test = training_data[0:1]
        
testing_data = pd.read_csv(
        "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/test_vectorised.csv")
testing_name = ("test_vectorised_shrunk")

def show_size(pd_object, name):
	
	if isinstance(pd_object,pd.DataFrame):
		usage_b = pd_object.memory_usage(deep=True).sum()
	else:
		usage_b = pd_object.memory_usage(deep=True)
	usage_mb = usage_b / (8 * (1024 ** 2)) # convert bits to megabytes
	return "{:03.2f} MB".format(usage_mb)

def reduce_data_int(csv_df, name):
	
	df_int = csv_df.select_dtypes(include=['int'])
	converted_int = df_int.apply(pd.to_numeric, downcast='unsigned')
	
	print("\nOrignial size:",show_size(df_int, name))
	print("\nReduced size:",show_size(converted_int, name))
	
	compared = pd.concat([df_int.dtypes,converted_int.dtypes],axis=1)
	compared.columns = ['before', 'after']
	print("\n",compared.apply(pd.Series.value_counts))
	
def reduce_object(csv_df, name):

	print("\nBefore reducing object:",csv_df.info())
	csv_df['comment_text'] = csv_df['comment_text'].apply(lambda content: content.split(','))
	#csv_df['comment_text'] = csv_df['comment_text'].apply(pd.to_numeric)
	#csv_df['comment_text'] = csv_df['comment_text'].apply(lambda array_float: array_float.to_numeric(downcast='float'))
	print("\nAfter reducing object:",csv_df.info())
	print(csv_df.head())

reduce_data_int(training_data,training_name)
reduce_data_int(testing_data,testing_name)

reduce_object(training_data_test,training_name)
#reduce_object(testing_data,testing_name)
