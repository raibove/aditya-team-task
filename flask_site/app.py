from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from nltk.corpus import stopwords
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from textblob import Word 
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import random


app = Flask(__name__)
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/question')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	no_sus = ["Happiness is where you are and satisfied with what you are","Good going","I am happy for you"]
	sus = ["Cheer up, there's more in life than what you think","Make it","Just talk"]


	Suicide = pd.read_csv('suicide.csv',encoding='ISO-8859-1')
	Suicide['Tweet'] = Suicide['Tweet'].fillna("")
	Suicide['lower_case']= Suicide['Tweet'].apply(lambda x: x.lower())      
	
	tokenizer = RegexpTokenizer(r'\w+')
	Suicide['Special_word'] = Suicide.apply(lambda row: tokenizer.tokenize(row['lower_case']), axis=1)    

	stop = stopwords.words('english')
	Suicide['stop_words'] = Suicide['Special_word'].apply(lambda x: [item for item in x if item not in stop])  

	Suicide['stop_words'] = Suicide['stop_words'].astype('str')        
	Suicide['string'] =Suicide['stop_words'].replace({"'": '', ',': ''}, regex=True)
	Suicide['string'] = Suicide['string'].str.findall('\w{3,}').str.join(' ') 

	#nltk.download('words')
	words = set(nltk.corpus.words.words())
	Suicide['NonEnglish'] = Suicide['string'].apply(lambda x: " ".join(x for x in x.split() if x in words))  

	Suicide['tweet'] = Suicide['NonEnglish'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 

	Suicide['label'] = Suicide['Suicide'].map({'Potential Suicide post':0,'Not Suicide post':1})
	X = Suicide['tweet']
	y = Suicide['label']
	all_word = []                              #set a dictionary for tweets and sentiment
	for i in range(len(Suicide)):
	    tweets = Suicide['tweet'][i]
	    sentiment= Suicide['Suicide'][i]
	    all_word.append((tweets,sentiment))
	

	tweets = []                                 #set dictionary after filtering the word
	for (words, sentiment) in all_word:
	    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
	    tweets.append((words_filtered, sentiment))

	def get_words_in_tweets(tweets):
	    all_words = []
	    for (words, sentiment) in tweets:
	        all_words.extend(words)
	    return all_words


	def get_word_features(wordlist):              #define in a wordlist
	    wordlist = nltk.FreqDist(wordlist)
	    word_features = wordlist.keys()
	    return word_features

	
	word_features = get_word_features(get_words_in_tweets(tweets))

	def extract_features(document):
		document_words = set(document)
		features = {}
		for word in word_features:
			features['contains(%s)' %word] = (word in document_words)
		return features
	training_set = nltk.classify.apply_features(extract_features, tweets) 
	classifier = nltk.NaiveBayesClassifier.train(training_set)  

	if request.method == 'POST':
		message = request.form['Que']
		my_prediction = classifier.classify(extract_features(message.split()))
		print(my_prediction)
	if my_prediction == "Potential Suicide post ":
		resp = random.choice(sus)
	else:
		resp = random.choice(no_sus)
	return render_template('result.html',prediction=my_prediction,response=resp)

if __name__ == '__main__':
	app.run(debug=True)