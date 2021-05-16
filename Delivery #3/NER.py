import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags, ne_chunk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import seaborn as sns
import pandas as pd

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('words')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
words = set(nltk.corpus.words.words())
nlp = en_core_web_sm.load()


def analyze_dataset(paths):
    # Creating dataset
    name = getVarName(paths, globals()).replace('Paths', '')
    print(name)
    os.makedirs('charts\\' + name)
    wholeText = ""
    wholeNer = []
    for path in paths:
        for txtFile in os.listdir(path):

            f = open(path + "\\" + txtFile, "r")
            text = f.read()
            while "  " in text:
                text.replace("  ", " ")

            new_string = re.sub('[^a-zA-Z0-9]', ' ', text)
            text = re.sub('\s+', ' ', new_string)

            indiNer = nlp(text)
            indiLabels = [x.label_ for x in indiNer.ents]
            indiLabelsCount = dict(Counter(indiLabels))

            if 'Fake' in path or 'deceptive' in path:
                indiLabelsCount['Result'] = 0
            elif 'True' in path or 'truthful' in path:
                indiLabelsCount['Result'] = 1

            wholeNer.append(indiLabelsCount)
            wholeText += text + " "

    # Create scatter plot matrix
    create_scatter(name, wholeNer)

    '''
    # Processing dataset
    iobTagged = process_ds(wholeText)
    BagOWords = bag_of_words(iobTagged)

    # Spacy
    ner = nlp(wholeText)
    labels = [x.label_ for x in ner.ents]
    labelsCount = dict(Counter(labels))

    # Creating graph
    create_graph(name, BagOWords)
    print_chart(name, 'NER', labelsCount)
    '''


def create_scatter(name, wholeNer):
    labels = ['NORP', 'CARDINAL', 'DATE', 'ORG', 'LANGUAGE', 'LOC', 'GPE', 'PERSON', 'ORDINAL', 'TIME', 'MONEY',
              'WORK_OF_ART', 'QUANTITY', 'FAC', 'PRODUCT', 'EVENT', 'LAW', 'PERCENT', 'Result']
    matrix = []

    for Ner in wholeNer:
        line = []
        for label in labels:
            if label in Ner.keys():
                line.append(Ner[label])
            else:
                line.append(0)
        matrix.append(line)

    df = pd.DataFrame(matrix, columns=labels)
    sns_plot = sns.pairplot(df, hue="Result")
    sns_plot.savefig('charts\\' + name + '\\NER_scatterPlot.png')


def process_ds(wholeText):
    # NLTK
    # Tokenize
    wholeTextArr = nltk.word_tokenize(wholeText)
    # Remove Stopwords
    wholeTextArrWOStopwords = remove_stopwords(wholeTextArr)
    # PoS tagging
    # see PoS tags : https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    PoSTagged = nltk.pos_tag(wholeTextArrWOStopwords)
    # Chunking
    chunked = chunking(PoSTagged)
    # IoB tagging
    # B-{TAG} : beginning of a phrase
    # I-{TAG} : describes that the word is inside of the current phrase
    # O : end of the sentence
    iobTagged = tree2conlltags(chunked)

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
    PoSFeatures = {}
    IoBFeatures = {}

    for data in dataset:
        PoSFeature = (data[0])[1]
        if PoSFeature not in PoSFeatures.keys():
            PoSFeatures[PoSFeature] = 1
        else:
            PoSFeatures[PoSFeature] = PoSFeatures[PoSFeature] + 1

        IoBFeature = (data[0])[2]
        if IoBFeature not in IoBFeatures.keys():
            IoBFeatures[IoBFeature] = 1
        else:
            IoBFeatures[IoBFeature] = IoBFeatures[IoBFeature] + 1

    print_chart(name, 'PoSFeatures', PoSFeatures)
    print_chart(name, 'IoBFeatures', IoBFeatures)


def print_chart(name, featureType, datasetDict):
    plt.bar(datasetDict.keys(), datasetDict.values())
    plt.xticks(rotation=90)
    plt.title(name + ' ' + featureType)
    plt.xlabel('Features')
    plt.ylabel('Count')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('charts\\' + name + '\\' + featureType + '.png')
    plt.close()


def getVarName(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]


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
