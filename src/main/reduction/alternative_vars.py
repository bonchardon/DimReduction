import pickle
import pandas as pd

import keras.models

import xgboost

from gensim import models
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence
import gensim.downloader
import gensim.downloader as api
from gensim.test.utils import get_tmpfile, datapath
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np
import nltk
import matplotlib.pyplot as plt

import plotly

import plotly.graph_objs as go

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

from argparse import Namespace
from functools import partial
from pathlib import Path

import math
import os
import pickle

# from pypi
# from dotenv import load_dotenv
from expects import (
    be_true,
    equal,
    expect,
)
from numpy.random import default_rng
from sklearn.decomposition import PCA

import holoviews
import hvplot.pandas
import numpy
import pandas

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
        from sklearn.decomposition import PCA

        word2vec = Word2Vec(cleaned_data, min_count=1, window=2)
        word2vec.train(cleaned_data, total_examples=1, epochs=1)

        # in case we want to save and use this later
        # word2vec.save('word2vec.model')
        # model = Word2Vec.load("word2vec.model")
        return word2vec

    def word2vec_pre(self):

        model = api.load('word2vec-google-news-300')
        # model.save('word2vec_pre.model')
        vector = model.wv['computer']
        return vector

    def doc2vec(self, data):

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]
        model = Doc2Vec(documents, vector_size=20, window=2, min_count=1, workers=4)
        # here is the way how to check similarity between docs

        test_doc = nltk.word_tokenize("random bullshit to check the effectiveness of the algo duh")

        return model.docvecs.most_similar(positive=[model.infer_vector(test_doc)], topn=5)

    def glove(self):

        glove_file = datapath("C:\/Users\Maryna Boroda\Documents\GitHub\DimReduction\models\glove.6B.50d.txt")
        word2vec_glove_file = get_tmpfile("glove.6B.50d.word2vec.txt")
        glove2word2vec(glove_file, word2vec_glove_file)

        model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

        filename = 'glove_word2vec.sav'
        pickle.dump(model, open(filename, 'wb'))

        # working with decorators


    def fastText(self, word_tokenized_corpus):
        """
        FastText algo may be useful, since with it's help one can get a vector representation of a word
        that is not in a corpus
        """

        embedding_size = 5
        window_size = 5
        min_word = 5
        down_sampling = 5

        ft_model = FastText(word_tokenized_corpus,
                            size=embedding_size,
                            window=window_size,
                            min_count=min_word,
                            sample=down_sampling,
                            sg=1,
                            iter=100)


class pca:

    @staticmethod
    def compute_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Calculate the principal components for X

        Args:
           X: of dimension (m,n) where each row corresponds to a word vector
           n_components: Number of components you want to keep.

        Return:
           X_reduced: data transformed in 2 dims/columns + regenerated original data
        """
        # you need to set axis to 0 or it will calculate the mean of the entire matrix instead of one per row
        X_demeaned = X - X.mean(axis=0)

        # calculate the covariance matrix
        # the default numpy.cov assumes the rows are variables, not columns so set rowvar to False
        covariance_matrix = np.cov(X_demeaned, rowvar=False)

        # calculate eigenvectors & eigenvalues of the covariance matrix
        eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)

        # sort eigenvalue in increasing order (get the indices from the sort)
        idx_sorted = np.argsort(eigen_vals)

        # reverse the order so that it's from highest to lowest.
        idx_sorted_decreasing = list(reversed(idx_sorted))

        # sort the eigen values by idx_sorted_decreasing
        eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]

        # sort eigenvectors using the idx_sorted_decreasing indices
        # We're only sorting the columns so remember to get all the rows in the slice
        eigen_vecs_sorted = eigen_vecs[:, idx_sorted_decreasing]

        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        # once again, make sure to get all the rows and only slice the columns
        eigen_vecs_subset = eigen_vecs_sorted[:, :n_components]

        # transform the data by multiplying the transpose of the eigenvectors
        # with the transpose of the de-meaned data
        # Then take the transpose of that product.
        X_reduced = np.dot(eigen_vecs_subset.T, X_demeaned.T).T
        return X_reduced


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import plotly.express as px

    vars = Alternative()

    words = PreProcessing.txt_preprocess(
            file_link=
            "C:\/Users\Maryna Boroda\Documents\GitHub\DimReduction\exampl_text\wikitext1.txt")

    words = [sent.split() for sent in words]
    #  print(vars.glove())

    model = pickle.load(open('glove_word2vec.sav', 'rb'))

    def append_list(sim_wrds, wrds):
        list_of_words = []
        for i in range(len(sim_wrds)):
            sim_words_list = list(sim_wrds[i])
            sim_words_list.append(words)
            sim_words_tuple = tuple(sim_words_list)
            list_of_words.append(sim_words_tuple)


    input_word = 'anarchism'
    user_input = model.most_similar(input_word, topn=10)
    words_try = ['anarchy', 'oil', 'gas', 'happy', 'sad', 'city', 'town',
             'village', 'country', 'continent', 'petroleum', 'joyful']

    subset = np.array([model[word] for word in words])
    reduced = pca.compute_pca(subset, n_components=2)
    reduced = pd.DataFrame(reduced, columns= 'X Y'.split())
    #reduced['Word'] = words_try
    print(reduced)

    # to visualize all the principal components
    fig = px.scatter_matrix(reduced, labels=words_try)
    fig.update_traces(diagonal_visible=False)
    fig.show()

    # 2D PCA scatter plot
    fig1 = px.scatter(reduced, color=words)
    fig1.show()


