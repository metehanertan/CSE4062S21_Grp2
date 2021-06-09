import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def analyze_dataset(paths):
    # Creating dataset
    name = getVarName(paths, globals()).replace('Paths', '')
    print(name)
    os.makedirs('charts\\' + name)
    dataset = []
    for path in paths:
        for txtFile in os.listdir(path):
            f = open(path + "\\" + txtFile, "r")
            text = f.read()
            while "  " in text:
                text.replace("  ", " ")

            new_string = re.sub('[^a-zA-Z0-9]', ' ', text)
            text = re.sub('\s+', ' ', new_string)
            dataset.append(text)

    vectorizer = TfidfVectorizer(stop_words={'english'})
    X = vectorizer.fit_transform(dataset)

    Sum_of_squared_distances = []
    K = range(2, 10)
    for k in K:
        km = KMeans(n_clusters=k, max_iter=200, n_init=10)
        km = km.fit(X)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.savefig('charts\\' + name + '\\k-means-clustering.png')
    plt.close()
    print('charts\\' + name + '\\k-means-clustering.png')

    true_k = 7
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
    model.fit(X)
    labels = model.labels_
    wiki_cl = pd.DataFrame(list(zip(range(len(dataset)), labels)), columns=['id', 'cluster'])

    f = open('charts\\' + name + '\\clusters.txt', "w")
    f.write(str(wiki_cl.sort_values(by=['cluster'])))
    f.close()
    print('charts\\' + name + '\\clusters.txt')

    result = {'cluster': labels, 'ds': dataset}
    result = pd.DataFrame(result)
    for k in range(0, true_k):
        s = result[result.cluster == k]
        text = s['ds'].str.cat(sep=' ')
        text = text.lower()
        text = ' '.join([word for word in text.split()])
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig('charts\\' + name + '\\cluster'+str(k)+'-wordcloud.png')
        plt.close()
        print('charts\\' + name + '\\cluster'+str(k)+'-wordcloud.png')


def getVarName(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]


# Yelp Dataset Paths
YelpDatasetPaths = ["..\\Datasets\\Fake",
                    "..\\Datasets\\True",
                    "..\\Datasets\\Testing"]

# Deceptive Opinion Spam Corpus v1.4 Paths
TurkDatasetPaths = ["..\\Datasets\\negative_polarity\\deceptive\\fold",
                    "..\\Datasets\\negative_polarity\\truthful\\fold",
                    "..\\Datasets\\positive_polarity\\deceptive\\fold",
                    "..\\Datasets\\positive_polarity\\truthful\\fold"]

analyze_dataset(YelpDatasetPaths)
analyze_dataset(TurkDatasetPaths)
