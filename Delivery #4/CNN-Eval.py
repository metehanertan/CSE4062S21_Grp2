import string
import re
from os import listdir
import os
import sys
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

if sys.argv[1] == "-yelp":
    # constants
    classA_path = '..\\Datasets\\True'
    classB_path = '..\\Datasets\\Fake'
    model_path = 'yelp.h5'
    testPath = "..\\Datasets\\TESTING"


elif sys.argv[1] == "-turk":
    # constants
    classA_path = '..\\Datasets\\negative_polarity/truthful/fold'
    classB_path = '..\\Datasets\\negative_polarity/deceptive/fold'
    model_path = 'model.h5'
    testPath = '..\\Datasets\\positive_polarity/deceptive/fold'
    testPath2 = '..\\Datasets\\positive_polarity/truthful/fold'

# Word embeddings
vocabulary_path = 'vocab.txt'
classA = 'HAM'
classB = 'SPAM'


# load in memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read the doc
    try:
        text = file.read()
    except:
        os.remove(filename)
        return ''
    # close file
    file.close()
    return text


# doc to clean tokens
def clean_doc(doc, vocab):
    # doc splitted to tokens in white space
    tokens = doc.split()
    # regex for filtering characters
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # eliminate punctuation
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove words that where not included in vocabulary
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens


# load all the documents in the specified directory and process them
def process_docs(directory, vocab):
    documents = list()
    # for each file in the directory
    for filename in listdir(directory):
        # create path
        path = directory + '/' + filename
        # load the document
        doc = load_doc(path)
        # from doc to tokens
        tokens = clean_doc(doc, vocab)
        # filter too long documents (mora than 2000 tokens)
        if len(doc.split()) < 2000:
            documents.append(tokens)
    return documents


# load and clean the dataset
def load_clean_dataset(vocab, is_train):
    # load docs
    if is_train:
        classA = process_docs(classA_path, vocab)
        classB = process_docs(classB_path, vocab)
    docs = classA + classB
    # Assign labels
    labels = array([0 for _ in range(len(classA))] + [1 for _ in range(len(classB))])
    return docs, labels


# from text to token
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# code and fill documents
def encode_docs(tokenizer, max_length, docs):
    # integer codification
    encoded = tokenizer.texts_to_sequences(docs)
    # sequences to pads
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded


# classify a review as A or B classes
def predict_class(review, vocab, tokenizer, max_length, model):
    # review
    line = clean_doc(review, vocab)
    # code and pads review
    padded = encode_docs(tokenizer, max_length, [line])
    # predict class
    yhat = model.predict(padded, verbose=0)
    # retrive estimated percetaje and label
    percent_pos = yhat[0, 0]
    if round(percent_pos) == 0:
        return (1 - percent_pos), classA
    return percent_pos, classB


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
    plt.savefig('Results\\CNN' + sys.argv[1] + '-AUC.png')


# load the extracted vocabulary in training
vocab_filename = vocabulary_path
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())

# load all comments: True for selecting the ones in training dataset, False for the testing one
train_docs, ytrain = load_clean_dataset(vocab, True)

# creating the tokenizer keras instance
tokenizer = create_tokenizer(train_docs)
# define vocab size
vocab_size = len(tokenizer.word_index) + 1
print('* Vocabulary size: %d' % vocab_size)

# compute maximum length of each tokenize document. MUST be same as the infered in
# the training (model fit)
max_length = max([len(s.split()) for s in train_docs])
print('* Max length: %d' % max_length)

# from document to a vector
Xtrain = encode_docs(tokenizer, max_length, train_docs)

# load the model
model = load_model(model_path)

# TESTING PART
true = 0
y_pred = []
y_test = []
size = 0

if sys.argv[1] == "-yelp":
    f = open("Results/CNN-yelp.txt", "w")

    # creating y_test
    size = 200
    for i in range(int(size/2)):
        y_test.append(int(0))
    for i in range(int(size/2)):
        y_test.append(int(1))

    for data_file in os.listdir(testPath):
        text_A = load_doc(testPath + "/" + data_file)
        percent, classification = predict_class(text_A, vocab, tokenizer, max_length, model)
        # print(f'* {classification} name:\n{data_file}* --> {classification} with {percent * 100} scoring')
        if str(classification) == "SPAM" and "Fake" in str(data_file):
            true += 1
        elif str(classification) == "HAM" and "True" in str(data_file):
            true += 1

        if str(classification) == "SPAM":
            f.write("0")
            y_pred.append(int(0))
        elif str(classification) == "HAM":
            f.write("1")
            y_pred.append(int(1))

elif sys.argv[1] == "-turk":
    f = open("Results/CNN-turk.txt", "w")

    # creating y_test
    size = int(800)
    for i in range(int(size/2)):
        y_test.append(int(0))
    for i in range(int(size/2)):
        y_test.append(int(1))

    for data_file in os.listdir(testPath):
        text_A = load_doc(testPath + "/" + data_file)
        percent, classification = predict_class(text_A, vocab, tokenizer, max_length, model)
        # print(f'* {classification} name:\n{data_file}* --> {classification} with {percent * 100} scoring')
        if str(classification) == "SPAM" and "d_" in str(data_file):
            true += 1

        if str(classification) == "SPAM":
            f.write("0")
            y_pred.append(int(0))
        elif str(classification) == "HAM":
            f.write("1")
            y_pred.append(int(1))

    for data_file in os.listdir(testPath2):
        text_B = load_doc(testPath2 + "/" + data_file)
        percent, classification = predict_class(text_B, vocab, tokenizer, max_length, model)
        # print(f'* {classification} name:\n{data_file}* --> {classification} with {percent * 100} scoring')
        f.write(str(classification))
        if str(classification) == "HAM" and "t_" in str(data_file):
            true += 1

        if str(classification) == "SPAM":
            f.write("0")
            y_pred.append(int(0))
        elif str(classification) == "HAM":
            f.write("1")
            y_pred.append(int(1))

print("Accuracy: %.2f " % (true / size * 100))
print("Precision Score: ", precision_score(y_test, y_pred, average="macro"))
print("Recall Score: ", recall_score(y_test, y_pred, average="macro"))
print("F1 Score: ", f1_score(y_test, y_pred, average="macro"))
plot_roc_curve(y_test, y_pred)

f.close()
