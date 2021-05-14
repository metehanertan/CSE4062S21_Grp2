import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('words')
# nltk.download('averaged_perceptron_tagger')
words = set(nltk.corpus.words.words())


def analyze_dataset(paths):
    # Creating dataset
    wholeText = ""
    for path in paths:
        for txtFile in os.listdir(path):

            f = open(path + "\\" + txtFile, "r")
            text = f.read()
            while "  " in text:
                text.replace("  ", " ")

            new_string = re.sub('[^a-zA-Z0-9]', ' ', text)
            text = re.sub('\s+', ' ', new_string)

            wholeText += text + " "

    # Processing dataset
    iobTagged = process_ds(wholeText)
    BagOWords = bag_of_words(iobTagged)

    # Creating graph of top 20 words
    create_graph(getVarName(paths, globals()), BagOWords[:20])


def process_ds(wholeText):
    # Tokenize
    wholeTextArr = nltk.word_tokenize(wholeText)
    # Remove Stopwords
    wholeTextArr = remove_stopwords(wholeTextArr)
    # PoS tagging
    # noun phrase NP, determiner DT, adjectives JJ, Noun NN
    wholeTextArr = nltk.pos_tag(wholeTextArr)
    # Chunking
    cs = chunking(wholeTextArr)
    # IoB tagging
    iobTagged = tree2conlltags(cs)
    return iobTagged


def remove_stopwords(word_list):
    processed_word_list = []
    for word in word_list:
        word = word.lower()
        if word not in stopwords.words("english") and len(word) > 0:
            processed_word_list.append(word)
    return processed_word_list


def chunking(ds):
    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(ds)
    return cs


def bag_of_words(corpus):
    wordfreq = {}
    for token in corpus:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1
    sorted_x = sorted(wordfreq.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_x


def create_graph(name, dataset):
    xAxis = []
    yAxis = []
    for data in dataset:
        xAxis.append(str(data[0]))
        yAxis.append(data[1])

    plt.bar(xAxis, yAxis)
    plt.xticks(rotation=90)
    plt.title(name)
    plt.xlabel('Words')
    plt.ylabel('Values')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('charts\\' + name + '.png')
    plt.close()


def getVarName(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0].replace("Paths", "")


# Yelp Dataset Paths
YelpTrainPaths = ["..\\Datasets\\Fake", "..\\Datasets\\True"]
YelpTestPaths = ["..\\Datasets\\Testing"]
# Deceptive Opinion Spam Corpus v1.4 Paths
TurkTrainPaths = ["..\\Datasets\\negative_polarity\\deceptive\\fold", "..\\Datasets\\negative_polarity\\truthful\\fold"]
TurkTestPaths = ["..\\Datasets\\positive_polarity\\deceptive\\fold", "..\\Datasets\\positive_polarity\\truthful\\fold"]

# Yelp Dataset
analyze_dataset(YelpTrainPaths)
analyze_dataset(YelpTestPaths)
# Deceptive Opinion Spam Corpus v1.4
analyze_dataset(TurkTrainPaths)
analyze_dataset(TurkTestPaths)
