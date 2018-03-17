###Importing Modules

print("Importing Modules")
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from multiprocessing import Pool
print("Modules Imported\n")


####Training Data
print("TRAINING DATA\n")
print("Importing Train CSV File")

train_csv = pd.read_csv("/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/train.csv", low_memory = False) #loads data frame using pandas
train_name = ("train_tagged")

train = (train_csv, train_name)

print("Training data frames loaded \n")

####Testing Data
print("\nTESTING DATA")
print("Importing Test CSV File")

test_csv = pd.read_csv("/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/test.csv", low_memory = False) ##loads data frame using pandas
test_name = ("test_tagged")
test = (test_csv, test_name)

print("Testing data frames loaded \n")


def tag(csv, name):
	shape = csv.shape #dimensions of the data frame
	rows = int(shape[0]) - 1#taking the number of rows

	stop_words = set(stopwords.words('english')) ##uses nltk stop words (useless words)
	filtered_sentence = []
	processed =[]
	tagged = []
	tagged_len = int

	print("Text preprocessing started:")

	for i in range (0,10):         
	   
		temp = str(csv.loc[i, 'comment_text'])    ###defines temp as the cell
		temp_token = word_tokenize(temp)	###tokenizes (splits up sentence into words) words in temp
		filtered_sentence = [word for word in temp_token if word != stop_words] ###filtered sentence = every word in temp that is not in stop words is 
		
		tagged = str(nltk.pos_tag(filtered_sentence)) ###tags (categorizes) word in filtered sentence
		#tagged = [word for word in tagged word = unicode(str, errors="replace")]
		tagged_len = int(len(tagged))  ###counts the number of tagged words

		csv.at[i, 'comment_text'] = [tagged]  ###inserts list of tagged words into the same cell the original sentence came from
		csv.loc[i, 'tagged_length'] = tagged_len ###inserts length of tagged words into a empty column
		
		print("Rows completed: {} / {}    Progress {:2.1%}".format(i, rows, i /rows), end="\r") ###prints progress of loop

	print("\nText preprocessing finished")
	csv.to_csv("/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/{0}.csv".format(name), index = False)  ###writes modified csv to a csv file
	print("DATA TAGGING FINISHED")


p = Pool(processes=2)
p.starmap(tag, ([train , test]))

p.close()
p.join()
