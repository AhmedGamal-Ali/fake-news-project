import pandas as pd 
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import pickle

Fake_News=pd.read_csv(r"C:\Users\AFC\Data Science\project\Fake.csv")
True_News=pd.read_csv(r"C:\Users\AFC\Data Science\project\True.csv")
Fake_News['target']=0
True_News['target']=1
data=pd.concat([Fake_News,True_News], ignore_index=True, sort=False)
new_data=data.drop(columns=["date","title","subject"],axis=1)

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

new_data["text"]=new_data["text"].apply(cleaning)

X=data['text']
y=data['target']

#Tfidf vectorizer
tv=TfidfVectorizer()
#transformed train reviews
X=tv.fit_transform(X)


pickle.dump(tv, open('tranform.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X,y ,random_state=0)



dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

filename = 'nlp_model.pkl'
pickle.dump(dt, open(filename, 'wb'))


