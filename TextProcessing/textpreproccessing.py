# importing modules

from sys import getsizeof
import spacy
import pandas as pd

# loading spacy
nlp = spacy.load('en')

# importing training data
training_data_untagged = pd.read_csv(
    "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/train.csv")
train_name = ("train_tagged")

print("Training data has been loaded")

# importing testing data
testing_data_untagged = pd.read_csv(
    "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/test.csv")
test_name = ("test_tagged")

print("Testing data has been loaded")

# text processing


def tagging(csv, name):
    shape = csv.shape  # dimensions of the data frame
    rows = int(shape[0]) - 1  # gets number of rows

    for row in range(0, rows):

        # temp is a string from data frame
        cell_string = str(csv.loc[row, 'comment_text'])
        print("Cell_string:{0}".format(getsizeof(cell_string)))

        cell_string_tagged = nlp(cell_string)
        print("Cell_string_tagged:{0}".format(getsizeof(cell_string_tagged)))

        cell_tagged_len = int(len(cell_string_tagged))
        print("Cell_tagged_len:{0}".format(getsizeof(cell_tagged_len)))

        csv.at[row, 'comment_text'] = [cell_string_tagged]
        csv.loc[row, 'tagged_length'] = cell_tagged_len

        print("Memory usage is:{0}".format(csv.info(memory_usage='deep'), end="\r"))
        print("Rows completed: {} / {}    Progress {:2.1%}".format(row,
                                                                   rows, row / rows), end="\r")  # prints progress of processing
    csv.to_csv(
        "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/{0}.csv".format(name),
        index=False)  # writes csv to file



tagging(training_data_untagged, train_name)
tagging(testing_data_untagged, test_name)
