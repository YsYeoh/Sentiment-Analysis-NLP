import pandas as pd
from NLP import clean_text, stemmingAndRemoveStopWord, cv
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.text import FreqDistVisualizer

#Read the kaggle dataset
tweet_data = pd.read_csv('CSV/tweets_scrapped.csv')

#Pre-processing of the text
tweet_data["content"] = tweet_data["content"].apply(clean_text)
tweet_data["content"] = tweet_data["content"].apply(stemmingAndRemoveStopWord)

#Seperate Labels and Features
x = tweet_data["content"]
y = tweet_data["sentiment"]

x_cv = cv.transform(x)

def showEmotionCountDiagram():
    print("Total number of data: " + str(len(y)))
    print(pd.read_csv('CSV/tweets_scrapped.csv'))
    plt.bar(y.value_counts().index, list(y.value_counts()))
    for index, value in enumerate(list(y.value_counts())):
        plt.text(index, value+100, str(value))
    plt.title('Emotion Count')
    plt.show()

def showFrequencyCountDiagram():
    visualizer = FreqDistVisualizer(features=cv.get_feature_names(), orient='h', n=30)
    visualizer.fit(x_cv)
    visualizer.set_title("Frequency Distribution of Top 30 Words")
    visualizer.show()
