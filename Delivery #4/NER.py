import os
import re
import spacy
from matplotlib import pyplot as plt
from spacy import displacy
from collections import Counter
import en_core_web_sm

nlp = en_core_web_sm.load()


def getVarName(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]


def analyze_dataset(paths):
    # Creating dataset
    name = getVarName(paths, globals()).replace('Paths', '')
    print(name)
    TrueNER = {}
    FakeNER = {}
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

            if 'Fake' in path or 'deceptive' in path or 'Fake' in txtFile:
                for label, count in zip(indiLabelsCount.keys(), indiLabelsCount.values()):
                    if label in FakeNER.keys():
                        FakeNER[label] = FakeNER[label] + count
                    else:
                        FakeNER[label] = count
            elif 'True' in path or 'truthful' in path or 'True' in txtFile:
                for label, count in zip(indiLabelsCount.keys(), indiLabelsCount.values()):
                    if label in TrueNER.keys():
                        TrueNER[label] = TrueNER[label] + count
                    else:
                        TrueNER[label] = count

    sortedTrue = dict(sorted(TrueNER.items(), key=lambda item: item[1]))
    sortedFake = dict(sorted(FakeNER.items(), key=lambda item: item[1]))
    print_chart(name, sortedTrue, sortedFake)


def print_chart(name, sortedTrue, sortedFake):
    plt.bar(sortedTrue.keys(), sortedTrue.values(), color='green', edgecolor='darkgreen')
    plt.bar(sortedFake.keys(), sortedFake.values(), color='red', edgecolor='darkred')
    plt.xticks(rotation=90)
    plt.title(name + ' NER features ')
    plt.xlabel('Features')
    plt.ylabel('Count')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('Results\\' + name + '-NERFeatures.png')
    plt.close()


# Yelp Dataset Paths
YelpTrainPaths = ["..\\Datasets\\Fake", "..\\Datasets\\True"]
YelpTestPaths = ["..\\Datasets\\Testing"]
# Deceptive Opinion Spam Corpus v1.4 Paths
TurkTrainPaths = ["..\\Datasets\\negative_polarity\\deceptive\\fold", "..\\Datasets\\negative_polarity\\truthful\\fold"]
TurkTestPaths = ["..\\Datasets\\positive_polarity\\deceptive\\fold", "..\\Datasets\\positive_polarity\\truthful\\fold"]

analyze_dataset(YelpTrainPaths)
analyze_dataset(YelpTestPaths)
analyze_dataset(TurkTrainPaths)
analyze_dataset(TurkTestPaths)
