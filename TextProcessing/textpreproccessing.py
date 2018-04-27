# importing modules

from sys import getsizeof
import spacy
import pandas as pd

# loading spacy
nlp = spacy.load('en_core_web_lg')

# importing training data
training_data_untagged = pd.read_csv(
    "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/train.csv")
train_name = ("train_tagged")

print("\nTraining data has been loaded")

# importing testing data
testing_data_untagged = pd.read_csv(
    "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/test.csv")
test_name = ("test_tagged")

print("\nTesting data has been loaded")

# text processing


def tagging(csv, name):
    shape = csv.shape  # dimensions of the data frame
    rows = int(shape[0]) - 1  # gets number of rows
    start = 0
    end = 5000
    completed = False

    while completed != True:

        new_csv = pd.DataFrame(columns=['word_vectors', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

        for row in range(start, end):

            comment_text = str(csv.loc[row, 'comment_text'])
            toxic = int(csv.loc[row, 'toxic'])
            severe_toxic = int(csv.loc[row, 'severe_toxic'])
            obscene = int(csv.loc[row, 'obscene'])
            threat = int(csv.loc[row, 'threat'])
            insult = int(csv.loc[row, 'insult'])
            identity_hate = int(csv.loc[row, 'identity_hate'])
            
            comment_text = nlp(comment_text)

            comment_text_vectorised = comment_text.vector

            new_csv.at[row, 'word_vectors'] = [comment_text_vectorised]
            new_csv.at[row, 'toxic'] = [toxic]
            new_csv.at[row, 'severe_toxic'] = [severe_toxic]
            new_csv.at[row, 'obscene'] = [obscene]
            new_csv.at[row, 'threat'] = [threat]
            new_csv.at[row, 'insult'] = [insult]
            new_csv.at[row, 'identity_hate'] = [identity_hate]

            print("Rows completed: {} / {}    Progress {:2.1%}".format((row), end,
                                                                       (row - start) / (end - start), end="\r"))  # prints progress of processing

        new_csv.to_csv(
            "/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/{0}/{0}_rows_{1}_to_{2}.csv".format(
                name,
                start,
                end),
            index=False)  # writes csv to file

        print("\nRows {0} to {1} completed".format(start, end))

        start = end
        end += 5000
        
        if start == rows:
            completed = True

        if end > rows:
            end = rows
        




tagging(training_data_untagged, train_name)
tagging(testing_data_untagged, test_name)
