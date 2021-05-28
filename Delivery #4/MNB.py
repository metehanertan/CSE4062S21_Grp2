import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import sys
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


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
    plt.savefig('Results\\MNB' + sys.argv[1] + '-AUC.png')


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

# Initialising the lists in which the polarity, review and either it's fake or true will be stored
polarity_class = []
reviews = []
spamity_class = []

reviews_test = []
pos_list = []

# yelp
if sys.argv[1] == "-yelp":
    for data_file in sorted(os.listdir(yelp_fake)):
        spamity_class.append("d")
        with open(os.path.join(yelp_fake, data_file)) as f:
            contents = f.read()
            reviews.append(contents)
    for data_file in sorted(os.listdir(yelp_true)):
        spamity_class.append("t")
        with open(os.path.join(yelp_true, data_file)) as f:
            contents = f.read()
            reviews.append(contents)
elif sys.argv[1] == "-turk":
    for data_file in sorted(os.listdir(insidendec)):
        polarity_class.append('negative')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insidendec, data_file)) as f:
            contents = f.read()
            reviews.append(contents)
    for data_file in sorted(os.listdir(insidentru)):
        polarity_class.append('negative')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insidentru, data_file)) as f:
            contents = f.read()
            reviews.append(contents)
    for data_file in sorted(os.listdir(insidepdec)):
        polarity_class.append('positive')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insidepdec, data_file)) as f:
            contents = f.read()
            reviews.append(contents)
    for data_file in sorted(os.listdir(insideptru)):
        polarity_class.append('positive')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insideptru, data_file)) as f:
            contents = f.read()
            reviews.append(contents)

# Making the dataframe using pandas to store polarity, reviews and true or fake

# Setting '0' for deceptive review and '1' for true review


# data_fm = pd.DataFrame({'polarity_class':polarity_class,'review':reviews,'spamity_class':spamity_class})
data_fm = pd.DataFrame({'review': reviews, 'spamity_class': spamity_class})

data_fm.loc[data_fm['spamity_class'] == 'd', 'spamity_class'] = 0
data_fm.loc[data_fm['spamity_class'] == 't', 'spamity_class'] = 1

# TESTING PART
file_name = []
for data_file in os.listdir(testPath):
    with open(os.path.join(testPath, data_file)) as f:
        contents = f.read()
        reviews_test.append(contents)
        file_name.append(data_file)

data_fm_test = pd.DataFrame({'review': reviews_test})

data_test_x = data_fm_test['review']

# Splitting the dataset to training and testing (0.7 and 0.3)
data_x = data_fm['review']
data_y = np.asarray(data_fm['spamity_class'], dtype=int)
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

# Using CountVectorizer() method to extract features from the text reviews and convert it to numeric data, which can be used to train the classifier

# Using fit_transform() for X_train and only using transform() for X_test

cv = CountVectorizer()

X_traincv = cv.fit_transform(X_train)
X_testcv = cv.transform(X_test)

data_test_x_cv = cv.transform(data_test_x)
# Using Naive Bayes Multinomial method as the classifier and training the data
nbayes = MultinomialNB()
nbayes.fit(X_traincv, y_train)

# Predicting the fake or deceptive reviews

# using X_testcv : which is vectorized such that the dimensions are matched
y_predictions = nbayes.predict(X_testcv)
y_r = nbayes.predict(data_test_x_cv)
# print(y_r)

f = open("Results/MNB{}.txt".format(sys.argv[1]), "w")
for result in y_r:
    f.write(str(result))
f.close()

y_result_test = list(y_r)
yres = ["True" if a == 1 else "Deceptive" for a in y_result_test]
i = 0
for p in yres:
    # print(file_name[i], " Review is", p)
    i += 1
# Printing out fake or deceptive reviews

# In[10]:

y_result = list(y_predictions)
yp = ["True" if a == 1 else "Deceptive" for a in y_result]
X_testlist = list(X_test)
output_fm = pd.DataFrame({'Review': X_testlist, 'True(1)/Deceptive(0)': yp})

# prepare the cross-validation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=6, random_state=10)
scores = cross_val_score(nbayes, X_traincv, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('K fold Scores: ', scores)

print("Accuracy: %.2f " % (metrics.accuracy_score(y_test, y_predictions) * 100))
print("Precision Score: ", precision_score(y_test, y_predictions, average='micro'))
print("Recall Score: ", recall_score(y_test, y_predictions, average='micro'))
print("F1 Score: ", f1_score(y_test, y_predictions, average='micro'))
plot_roc_curve(y_test, y_predictions)
