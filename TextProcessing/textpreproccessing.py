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

# importing testing data
testing_data_untagged = pd.read_csv(
    "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/test.csv")
test_name = ("test_vectorised")

print("\nTesting data has been loaded")

#tagging(training_data_untagged, train_name)
#tagging(testing_data_untagged, test_name)

def iterate(csv_df, name):
    rows = int(csv_df.shape[0])  # gets number of rows
    start = 0
    end = rows
    completed = False
    print(rows)

    while completed != True:

        df = csv_df[start:rows]

        df['comment_text'] = df['comment_text'].apply(lambda text: (nlp(str(text))).vector) #vectorises text
        print("\n",df.head())
        print("\ndf.head",csv_df.head())
        print("\ndf.tail",csv_df.tail())

        print("\nRows {0} to {1} completed\n".format(start, end))

        start = end
        end += 3000
        
        if start == rows:
            completed = True #ends once all rows are completed

        if end > rows:
            end = rows

    csv_df.to_csv(
            "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/{0}.csv".format(
                name,
                str(start),
                str(end)),
            index=False)  # writes csv to file

iterate(training_data_untagged, train_name)
iterate(testing_data_untagged, test_name)
