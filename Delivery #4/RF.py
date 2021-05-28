import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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


def plot_roc_curve(y_true, y_score, size=None):
    """plot_roc_curve."""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_true, y_score)
    if size is not None:
        plt.figure(figsize=(size, size))
        plt.axis('equal')
    plt.plot(false_positive_rate, true_positive_rate, lw=2, color='navy')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.grid()
    plt.title('Receiver operating characteristic AUC={0:0.2f}'.format(
        roc_auc_score(y_true, y_score)))
    plt.savefig('Results\\RF' + sys.argv[1] + '-AUC.png')


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

nb = RandomForestClassifier()
nb.fit(x_train, y_train)
pred_3 = nb.predict(cv.transform(x_test))

# prepare the cross-validation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=6, random_state=10)
scores = cross_val_score(nb, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('K fold Scores: ', scores)

print("Accuracy: %.2f " % (metrics.accuracy_score(y_test, pred_3) * 100))
print("Precision Score: ", precision_score(y_test, pred_3, average='micro'))
print("Recall Score: ", recall_score(y_test, pred_3, average='micro'))
print("F1 Score: ", f1_score(y_test, pred_3, average='micro'))
plot_roc_curve(y_test, pred_3)
