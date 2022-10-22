# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:41:49 2022

@author: AFC
"""
from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import pickle

#Removal of HTML Contents
def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removal of Punctuation Marks
def remove_punctuations(text):
    return re.sub('\[[^]]*\]', '', text)

# Removal of Special Characters
def remove_special_characters(text):
    return re.sub("[^a-zA-Z]"," ",text)

stop_words=set(stopwords.words("english"))
lemma = nltk.WordNetLemmatizer()


def remove_stopwords_and_lemmatization(text):
    final_text = []
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    for word in text:
        if word not in stop_words:
            word = lemma.lemmatize(word) 
            final_text.append(word)
    return " ".join(final_text)

def cleaning(text):
    text=remove_html(text)
    text=remove_punctuations(text)
    text=remove_special_characters(text)
    text=remove_stopwords_and_lemmatization(text)
    return text

# load the model from disk
filename = 'nlp_model.pkl'
dt = pickle.load(open(filename, 'rb'))
tv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        #message = request.args.get('message')
        cleaning(message)
        data = [message]
        vect = tv.transform(data).toarray()
        my_prediction = dt.predict(vect)
        print("my_prediction: " ,my_prediction)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)

