import pandas as pd 
import dask.dataframe as dd
import regex as re
import numpy as np

train_path = "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/train.csv"
test_path = "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

def clean(df,file_path):
	
	df['comment_text'] = df['comment_text'].apply(lambda remove: re.sub('\p{P}+','', str(remove))) #removes punctuation from string
	df['comment_text'] = df['comment_text'].apply(lambda remove: re.sub('\u007C', '', str(remove)))
	df['comment_text'] = df['comment_text'].apply(lambda up: up.upper())

	df = df.replace('', np.nan, regex=True)
	df = df.dropna()
	df['id'] = df['id'].astype('int')
	
	try: ##removes columns that can be generated in error - cause should be fixed
		df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1',  'Unnamed: 0.1.1',  'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1'  'Unnamed: 0.1.1.1.1.1'  'Unnamed: 0.1.1.1.1.1.1','Unnamed: 0.1.1.1.1.1.1.1'])
	
	except ValueError:
		print("ValueError")
		pass

	df.to_csv(file_path, index=False)

clean(train, train_path)
clean(test, test_path)

print(train.head())   
print(test.head())

