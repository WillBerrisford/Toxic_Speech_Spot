# importing modules
import spacy
import pandas as pd

# loading spacy
from typing import Any

nlp = spacy.load('en_core_web_lg')

# importing training data
training_data_untagged = pd.read_csv(
	"/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/train.csv")
train_name = ("train_vectorised")
print(training_data_untagged.shape)

print("\nTraining data has been loaded")

def iterate(csv_df, name):
	rows = int(csv_df.shape[0])  # gets number of rows
	start = 0
	end = 1
	completed = False
	print(rows)

	df = csv_df[0:1]

	df['comment_text'] = df['comment_text'].apply(lambda text: nlp(str(text)).vector) #vectorises text
	print("\ndf.head:",df.head())
	print("\ndf.info:",df.info())
	print("\ndf.head",csv_df.head())
	print("\ndf.tail",csv_df.tail())

	csv_df.to_csv(
		"/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/{0}_test.csv".format(
			name,
			str(start),
			str(end)),
		index=False)  # writes csv to file

iterate(training_data_untagged, train_name)

