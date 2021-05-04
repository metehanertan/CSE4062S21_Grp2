import os
import nltk
import heapq
import numpy as np
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import re
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('words')
words = set(nltk.corpus.words.words())

def calculate_dataset_details(paths):
    wholeText = ""
    numberOfDocuments = 0
    minLength = 9999999
    maxLength = 0
    minWordCount = 9999999
    maxWordCount = 0
    for path in paths:
        for txtFile in os.listdir(path):
            numberOfDocuments += 1

            f = open(path+"\\"+txtFile, "r")
            text = f.read()
            while "  " in text:
                text.replace("  ", " ")

            new_string = re.sub('[^a-zA-Z0-9]', ' ', text)
            text = re.sub('\s+', ' ', new_string)

            wholeText += text + " "
            tokenizer = RegexpTokenizer(r'\w+')
            textArr = tokenizer.tokenize(text)

            if len(textArr) < minWordCount:
                minWordCount = len(textArr)
            elif len(textArr) > maxWordCount:
                maxWordCount = len(textArr)

            if len(text) < minLength:
                minLength = len(text)
            elif len(text) > maxLength:
                maxLength = len(text)

    wholeTextArr = wholeText.split(" ")
    print("--------------------------------------------")
    print("Dataset: ", namestr(paths, globals()))
    print("--------------------------------------------")
    print("Number of documents: ", numberOfDocuments)
    print("Number of words: ", len(wholeTextArr))
    print("Number of characters: ", len(wholeText))
    print("Min word count: ", minWordCount)
    print("Average word count: ", len(wholeTextArr)/numberOfDocuments)
    print("Max word count: ", maxWordCount)
    print("Min character count: ", minLength)
    print("Average character count: ", len(wholeText)/numberOfDocuments)
    print("Max character count: ", maxLength)

    find_bag_of_words(wholeTextArr)


def find_bag_of_words(corpus):
    print("Bag of words top 10: ")
    bag_of_words(corpus)
    print("Bag of words without stopwords top 10: ")
    bag_of_words(remove_stopwords(corpus))


def bag_of_words(corpus):
    wordfreq = {}
    for token in corpus:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1
    sorted_x = sorted(wordfreq.items(), key=lambda kv: kv[1], reverse=True)
    print("Size of bag of words is ", len(sorted_x), ", top 10 words are:")
    print(sorted_x[0:10])


def remove_stopwords(word_list):
    processed_word_list = []
    for word in word_list:
        word = word.lower()
        if word not in stopwords.words("english") and len(word) > 0:
            processed_word_list.append(word)
    return processed_word_list


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0].replace("Paths", "")


# Yelp Dataset
YelpTrainPaths = ["..\\Datasets\\Fake", "..\\Datasets\\True"]
YelpTestPaths = ["..\\Datasets\\Testing"]
# Deceptive Opinion Spam Corpus v1.4
TurkTrainPaths = ["..\\Datasets\\negative_polarity\\deceptive\\fold", "..\\Datasets\\negative_polarity\\truthful\\fold"]
TurkTestPaths = ["..\\Datasets\\positive_polarity\\deceptive\\fold", "..\\Datasets\\positive_polarity\\truthful\\fold"]

# Exploring Datasets
calculate_dataset_details(YelpTrainPaths)
calculate_dataset_details(YelpTestPaths)
calculate_dataset_details(TurkTrainPaths)
calculate_dataset_details(TurkTestPaths)
