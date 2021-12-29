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


class FeatureSelection:

    def __init__(self):
        pass

        """
        Types of feature selection for dim reduction:

            (1) recursive feature elimination;
            (2) genetic feature selection;
            (3) sequential feature selection (forward/backward).
        """

    def feature_elimination(self, data):

        """
        RFE works by searching for a subset of features by starting with all features in the training dataset
        and successfully removing features until the desired number remains.

        This is achieved by fitting the given machine learning algorithm used in the core of the model,
        ranking features by importance, discarding the least important features, and re-fitting the model.
        This process is repeated until a specified number of features remains.
        """

        # TODO: figure out how to split the data effectively into X, y values for further use

        X, y = None

        # There might be useful to drop highly correlated features by making a correlating matrix.
        correlated_features = set()
        correlation_matr = data.corr()

        for i in range(len(correlation_matr.columns)):
            for j in range(i):
                if abs(correlation_matr.iloc[i, j]) > 0.8:
                    colname = correlation_matr.columns[i]
                    correlated_features.add(colname)

        # if use RFECV, the number of features to select will be chosen automatically
        rfecv = RFECV(estimator=DecisionTreeClassifier(), step=1, cv=StratifiedKFold(10), scoring='accuracy')
        rfecv.fit(X, y)

    def feature_selection(self, data):

        """
        Genetic algorithms mimic the process of natural selection
        to search for optimal values of a function.
        """
        X, y = None
        estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
        selector = GeneticSelectionCV(estimator,
                                      cv=5,
                                      verbose=1,
                                      scoring="accuracy",
                                      max_features=5,
                                      n_population=50,
                                      crossover_proba=0.5,
                                      mutation_proba=0.2,
                                      n_generations=40,
                                      crossover_independent_proba=0.5,
                                      mutation_independent_proba=0.05,
                                      tournament_size=3,
                                      n_gen_no_change=10,
                                      caching=True,
                                      n_jobs=-1)
        selector = selector.fit_transform(X)

    def seq_feature_selection(self, data):

        """
        Some of the common techniques used for feature selection
        includes regularization techniques (L1 / L2 norm) and sequential forward / backward feature selection.

        The transformer that performs Sequential Feature Selection.

        "The Sequential Feature Selection adds (forward selection)
        or removes (backward selection) features to form a feature subset in a greedy fashion.
        At each stage, this estimator chooses the best feature to add or remove based on the cross-validation score of an estimator.
        In the case of unsupervised learning, this Sequential Feature Selector looks only at the features (X),
        not the desired outputs (y)."
        """
        X, y = data
        lr = LogisticRegression(C=1.0, random_state=1)
        knn = KNeighborsClassifier(n_neighbors=3)
        sfs = SequentialFeatureSelector(knn, n_features_to_select=100)
        sfs.fit_transform(X, y)
        return sfs.get_feature_names_out()


if __name__ == "__main__":

    select = FeatureSelection()
