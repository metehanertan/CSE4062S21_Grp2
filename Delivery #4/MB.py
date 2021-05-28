import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import re
import string

import sys

if sys.argv[1] == "-yelp":
    # yelp dataset
    yelp_fake = '..\\Datasets\\Fake'
    yelp_true = '..\\Datasets\\True'

elif sys.argv[1] == "-turk":
    insidendec = '..\\Datasets\\negative_polarity\\deceptive\\fold'
    insidentru = '..\\Datasets\\negative_polarity\\truthful\\fold'
    insidepdec = '..\\Datasets\\positive_polarity\\deceptive\\fold'
    insideptru = '..\\Datasets\\positive_polarity\\truthful\\fold'

testPath = '..\\Datasets\\TESTING'
polarity_class = []
reviews = []
spamity_class = []

reviews_test = []
pos_list = []

# yelp
if sys.argv[1] == "-yelp":
    for data_file in sorted(os.listdir(yelp_fake)):
        spamity_class.append(0)
        with open(os.path.join(yelp_fake, data_file)) as f:
            contents = f.read()
            reviews.append(contents)
    for data_file in sorted(os.listdir(yelp_true)):
        spamity_class.append(1)
        with open(os.path.join(yelp_true, data_file)) as f:
            contents = f.read()
            reviews.append(contents)
elif sys.argv[1] == "-turk":
    for data_file in sorted(os.listdir(insidendec)):
        polarity_class.append('negative')
        spamity_class.append(0)
        with open(os.path.join(insidendec, data_file)) as f:
            contents = f.read()
            reviews.append(contents)
    for data_file in sorted(os.listdir(insidentru)):
        polarity_class.append('negative')
        spamity_class.append(1)
        with open(os.path.join(insidentru, data_file)) as f:
            contents = f.read()
            reviews.append(contents)
    for data_file in sorted(os.listdir(insidepdec)):
        polarity_class.append('positive')
        spamity_class.append(1)
        with open(os.path.join(insidepdec, data_file)) as f:
            contents = f.read()
            reviews.append(contents)
    for data_file in sorted(os.listdir(insideptru)):
        polarity_class.append('positive')
        spamity_class.append(1)
        with open(os.path.join(insideptru, data_file)) as f:
            contents = f.read()
            reviews.append(contents)


def text_cleaning(text):
    '''
    Make text lowercase, remove text in square brackets,remove links,remove special characters
    and remove words containing numbers.
    '''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)  # remove special chars
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    return text


a = 0
for i in reviews:
    reviews[a] = text_cleaning(i)
    a += 1
data_fm = pd.DataFrame({'review': reviews, 'spamity_class': spamity_class})
data_x = data_fm['review']
data_y = np.asarray(data_fm['spamity_class'], dtype=int)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=0, test_size=0.2)
cv = CountVectorizer(ngram_range=(1, 2))
x_train = cv.fit_transform(x_train)

nb = MultinomialNB()
nb.fit(x_train, y_train)
pred_3 = nb.predict(cv.transform(x_test))
score_3 = accuracy_score(y_test, pred_3)
print(score_3)
