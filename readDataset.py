import pandas as pd
from NLP import clean_text, stemmingAndRemoveStopWord, cv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.text import FreqDistVisualizer
import numpy as np
np.random.seed(10)
remove_n = 600

lb = LabelEncoder()

#Read the kaggle dataset
tweet_data = pd.read_csv('CSV/emotion.csv')

#Too many fear sentiment, minus 600 row to fear
# drop_indices = np.random.choice(tweet_data[tweet_data['sentiment'].str.contains('fear')].index, remove_n, replace=False)
# tweet_data = tweet_data.drop(drop_indices)

#Pre-processing of the text
tweet_data["content"] = tweet_data["content"].apply(clean_text)
tweet_data["content"] = tweet_data["content"].apply(stemmingAndRemoveStopWord)

#Seperate Labels and Features
x = tweet_data["content"]
y = tweet_data["sentiment"]

#Split the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=48)

x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)

def showEmotionCountDiagram():
    print("Total number of data: " + str(len(y)))
    print(pd.read_csv('CSV/emotion.csv'))
    plt.bar(y.value_counts().index, list(y.value_counts()))
    for index, value in enumerate(list(y.value_counts())):
        plt.text(index, value+100, str(value))
    plt.title('Emotion Count')
    plt.show()

def showFrequencyCountDiagram():
    visualizer = FreqDistVisualizer(features=cv.get_feature_names(), orient='h', n=30)
    visualizer.fit(x_train_cv)
    visualizer.set_title("Frequency Distribution of Top 30 Words")
    visualizer.show()

def showTrainEmotionCountDiagram():
    plt.bar(y_train.value_counts().index, list(y_train.value_counts()))
    for index, value in enumerate(list(y_train.value_counts())):
        plt.text(index, value+100, str(value))
    plt.title('Train Emotion Count')
    plt.show()

def showTestEmotionCountDiagram():
    plt.bar(y_test.value_counts().index, list(y_test.value_counts()))
    for index, value in enumerate(list(y_test.value_counts())):
        plt.text(index, value+100, str(value))
    plt.title('Test Emotion Count')
    plt.show()


