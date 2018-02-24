###Importing Modules

print("Importing Modules")
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
print("Modules Imported\n")

def tag_train():
	
	print("TRAINING DATA\n")
	
	print("Importing Train CSV File")
	train_csv = pd.read_csv("/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/train.csv", low_memory = False) #loads data frame using pandas

	print("Training data frames loaded \n")

	shape = train_csv.shape #dimensions of the data frame
	rows = int(shape[0]) - 1#taking the number of rows

	stop_words = set(stopwords.words('english')) ##uses nltk stop words (useless words)
	filtered_sentence = []
	processed =[]
	tagged = []
	tagged_len = int

	print("Text preprocessing started (training):")

	for i in range (0,rows):         
	   
		temp = str(train_csv.loc[i, 'comment_text'])    ###defines temp as the cell
		temp_token = word_tokenize(temp)	###tokenizes (splits up sentence into words) words in temp
		filtered_sentence = [word for word in temp_token if word != stop_words] ###filtered sentence = every word in temp that is not in stop words is 
		
		tagged = str(nltk.pos_tag(filtered_sentence)) ###tags (categorizes) word in filtered sentence
		#tagged = [word for word in tagged word = unicode(str, errors="replace")]
		tagged_len = int(len(tagged))  ###counts the number of tagged words

		train_csv.at[i, 'comment_text'] = [tagged]  ###inserts list of tagged words into the same cell the original sentence came from
		train_csv.loc[i, 'tagged_length'] = tagged_len ###inserts length of tagged words into a empty column
		
		print("Rows completed: {} / {}    Progress {:2.1%}".format(i, rows, i /rows), end="\r") ###prints progress of loop

	print("\nText preprocessing finished")

	train_csv.to_csv("/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/train_tagged.csv", index = False)  ###writes modified csv to a csv file
	print("TRAINING DATA TAGGING FINISHED")
	
def tag_test():
	
	print("\nTESTING DATA")
	
	print("Importing Test CSV File")
	
	test_csv = pd.read_csv("/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/test.csv", low_memory = False) ##loads data frame using pandas

	print("Testing data frames loaded \n")

	shape = test_csv.shape #dimensions of the data frame
	rows = int(shape[0]) - 1 #taking the number of rows

	stop_words = set(stopwords.words('english')) ##uses nltk stop words (useless words)
	filtered_sentence = []
	processed =[]
	tagged = []
	tagged_len = int

	print("Text preprocessing started (testing):")

	for i in range (0,rows):         
		
		temp = str(test_csv.loc[i, 'comment_text'])    ###defines temp as the cell
		temp_token = word_tokenize(temp)	###tokenizes (splits up sentence into words) words in temp
		filtered_sentence = [word for word in temp_token if word != stop_words] ###fitered sentence = every word in temp that is not in stop words is 

		tagged = str(nltk.pos_tag(filtered_sentence)) ###tags (categorizes) word in filtered sentence
		tagged_len = int(len(tagged))  ###counts the number of tagged words

		test_csv.at[i, 'comment_text'] = [tagged]  ###inserts list of tagged words into the same cell the original sentence came from
		test_csv.loc[i, 'tagged_length'] = tagged_len ###inserts length of tagged words into a empty column
		
		print("Rows completed: {} / {}    Progress {:2.1%}".format(i, rows, i /rows), end="\r") ###prints progress of loop 

	print("\nText preprocessing finished")

	test_csv.to_csv("/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/test_tagged.csv", index = False)  ###writes modified csv to a csv file
	
	print("TESTING DATA TAGGING FINISHED")

tag_train()
tag_test()


