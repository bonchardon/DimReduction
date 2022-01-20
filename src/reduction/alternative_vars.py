import keras.models

import xgboost

from gensim import models
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import LineSentence
import gensim.downloader
import gensim.downloader as api

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

from src.pre_stage import PreProcessing


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
        word2vec = Word2Vec([i for i in cleaned_data], min_count=1)
        word2vec.build_vocab(cleaned_data)
        word2vec.save('word2vec.model')

        model = Word2Vec.load("word2vec.model")
        model.train([cleaned_data], total_examples=1, epochs=1)
        # sims = word2vec.wv.most_similar('anarchism', topn=2)
        vector = model.wv.key_to_index['anarchism']
        return vector

    def word2vec_pre(self):

        model = api.load('word2vec-google-news-300')
        # model.save('word2vec_pre.model')

        vector = model.wv['computer']
        return vector

    def doc2vec(self, data):
        from gensim.test.utils import get_tmpfile
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]
        model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

        file_ = get_tmpfile("doc2vec_model")
        model.save(file_)

        model = Doc2Vec.load(file_)

        vector = model.infer_vector([data])

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

    def fastText(self, cleaned_data):

        ft_model = FastText(word_tokenized_corpus,
                            size=embedding_size,
                            window=window_size,
                            min_count=min_word,
                            sample=down_sampling,
                            sg=1,
                            iter=100)


if __name__ == "__main__":

    vars = Alternative()
    words = PreProcessing.txt_preprocess(
            file_link=
            "C:\/Users\Maryna Boroda\Documents\GitHub\DimReduction\exampl_text\wikitext1.txt")

    # words = [sent.split() for sent in words]
    # words = words.split()
    # print(vars.word2vec(cleaned_data=words))

    print(vars.word2vec_pre)


