import pandas as pd 
import dask.dataframe as dd
import regex as re

train_path = "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/train.csv"
test_path = "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

def clean(df,file_path):
	
	df['comment_text'] = df['comment_text'].apply(lambda remove: re.sub('\p{P}+','', str(remove))) #removes punctuation from string
	df['comment_text'] = df['comment_text'].apply(lambda up: up.upper())  

	df.to_csv(file_path)

clean(train, train_path)
clean(test, test_path)

try: ##removes columns that can be generated in error ~ unsure of cause
	test = test.drop(columns=['Unnamed: 0'])
	train = train.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'])
	
except ValueError:
	pass

print(train.head())   
print(test.head())


