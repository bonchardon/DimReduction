import keras.models
from gensim import models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import spatial

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('stopwords')
# nltk.download('wordnet')

# from genetic_selection import GeneticSelectionCV
from sklearn import datasets, linear_model
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import Isomap
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE, RFECV
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import svm, datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, losses

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Dense, Reshape


class Alternative:

    """
    Other ways to reduce dimensionality is:

        1) try to train sparse model (like SGDClassifier with huge L1 penalty.
        (1.1.) it might help to transform word-count using TF-IDF before using the data in a linear classifier.
        (2) we can also use pre-trained dimensionality reducer, such as word2vec/fastText to extract features from text.
    """

    def __init__(self):
        pass

    def word2vec(self, cleaned_data):

        # TODO: prepare word2vec model for further work

        w2v = models.KeyedVectors.load_word2vec_format(
                            './GoogleNews-vectors-negative300.bin',
                            binary=True)
        custom_model = models.Word2Vec(cleaned_data,
                                       min_count=1, size=300,
                                       workers=4)

        # visualization part

        return custom_model

    def glove(self, cleaned_data):

        embeddings = {}

        with open('models/glove.6B.50d.txt', 'rb') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings[word] = vector


            # getting closest word by using glove algo
            def find_closest_embeddings(embedding):
                return sorted(embeddings.keys(), key=lambda word:

            spatial.distance.euclidean(embeddings[word], embedding))

            find_closest_embeddings(embeddings[b"example"])[:10]

    def fastText(self, cleaned_data):

        pass


if __name__ == "__main__":

    vars = Alternative()
