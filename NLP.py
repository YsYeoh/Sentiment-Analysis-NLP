import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer

#Stemming Import and stop word import
from nltk.corpus import stopwords
from porter2stemmer import Porter2Stemmer

stop = stopwords.words('english')
stemmer = Porter2Stemmer()
#cv = CountVectorizer(stop_words='english')
cv = TfidfVectorizer(stop_words='english')

def clean_text(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = text.lower()
    return text

def stemmingAndRemoveStopWord(text):
    str_ = ""
    for word in text.split():
        if word not in stop:
            word = stemmer.stem(word)
            str_ = str_ + " " + word
    return str_