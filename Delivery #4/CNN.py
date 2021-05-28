import re
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import os
import string
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import sys


if sys.argv[1] == "-yelp":
    # constants
    classA_path = '..\\Datasets\\True'
    classB_path = '..\\Datasets\\Fake'
    model_path = 'yelp.h5'


elif sys.argv[1] == "-turk":
    # constants
    classA_path = '..\\Datasets\\negative_polarity/truthful/fold'
    classB_path = '..\\Datasets\\negative_polarity/deceptive/fold'
    model_path = 'model.h5'



# Word embeddings
vocabulary_path = 'vocab.txt'
classA = 'HAM'
classB = 'SPAM'



min_occurrence = 5

# load doc in memory
def load_doc(filename):
    # open file as only read
    file = open(filename, 'r')
    # read text
    try:
        text = file.read()
    except:
        os.remove(filename)
        return ''
    # close file
    file.close()
    return text

# from doc to clean tokens
def clean_doc(doc):
    # dividido en tokens por espacio en blanco
    tokens = doc.split()
    # regex for character filter
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # filter punctuation
    tokens = [re_punc.sub('', w) for w in tokens]
    # delete tokens out of alphabetic order
    tokens = [word for word in tokens if word.isalpha()]
    # filter stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter tokens by size
    tokens = [word for word in tokens if len(word) < 15 and len(word) > 1]
    return tokens


# load doc and add to vocabulary
def add_doc_to_vocab(filename, vocab):
    # load odc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counter()
    vocab.update(tokens)

# load all documents in a directory
def process_docs(directory, vocab):
    # review all files in the directory
    for filename in listdir(directory):
        # path
        path = directory + '/' + filename
        # add vocabulary
        add_doc_to_vocab(path, vocab)


# save in file
def save_list(lines, filename):
    # convertir líneas a una sola nota de texto
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # cclose file
    file.close()


# define vocabulary
vocab = Counter()
vocab_A = Counter()
vocab_B = Counter()
# fill vocabulary from all the documents in the dataset
process_docs(classA_path, vocab_A)
frec_classA = str(vocab.most_common(50))
process_docs(classB_path, vocab_B)
# main words and vocbaulary size
print(f'* {classA} with vocabulary length {len(vocab_A)} as (word, frecuency): \n {vocab_A.most_common(75)} \n')
print(f'* {classB} with vocabulary length {len(vocab_B)} as (word, frecuency): \n {vocab_B.most_common(75)} \n')
vocab = vocab_A + vocab_B
print(f'* dataset with vocabulary length {len(vocab)} as (word, frecuency): \n {vocab.most_common(75)}')



# skip tokens under minimum occurrence
tokens = [k for k,c in vocab.items() if c >= min_occurrence]
print(f' From {len(vocab)} to {len(tokens)}: {len(vocab)-len(tokens)} words eliminated')

# save tokens into a vocabulary file
save_list(tokens, vocabulary_path)


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
def process_docs(directory, vocab, is_train):
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


# Loading and cleaning the dataset
def load_clean_dataset(vocab, is_train):
    # Merging documents into one variable; first class A samples, then class B samples.
    classA = process_docs(classA_path, vocab, is_train)
    classB = process_docs(classB_path, vocab, is_train)
    docs = classA + classB
    # Respective labels designated in the same order that the samples where loaded into docs.
    labels = array([0 for _ in range(len(classA))] + [1 for _ in range(len(classB))])
    return docs, labels


# creating a Keras tokenizer instance
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# code and fill documents
def encode_docs(tokenizer, max_length, docs):
    # integer codification
    encoded = tokenizer.texts_to_sequences(docs)
    # sequence to pad
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded


# define the model
def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model diagram
    model.summary()
    return model

# load the vocabulary
vocab = load_doc(vocabulary_path)
vocab = set(vocab.split())
print(vocab)

# Loading training data: first class A samples, then class B samples.
train_docs, ytrain = load_clean_dataset(vocab, True)
# class A
print(f'1st doc sample: {train_docs[0]} --> {ytrain[0]} \n')
# class B
print(f'Last doc sample: {train_docs[len(train_docs)-1]} --> {ytrain[len(train_docs)-1]}')

# Creating the tokenizer
tokenizer = create_tokenizer(train_docs)

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)
# For each tokenized document in docs, compute max sequence length
max_length = max([len(doc.split()) for doc in train_docs])
print('Max length: %d' % max_length)


Xtrain = encode_docs(tokenizer, max_length, train_docs)
model = define_model(vocab_size, max_length)
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
model.save(model_path)