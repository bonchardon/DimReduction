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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import datasets, linear_model
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
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

from src.pre_stage import PreProcessing, Vectorization
from src.distance_metrics import DistanceMetrics


class FeatureSelection:

    def __init__(self):
        pass

        """
        Types of feature selection for dim reduction:

            (1) recursive feature elimination;
            (2) genetic feature selection;
            (3) sequential feature selection (forward/backward).
        """

    def feature_elimination(self, X, y):
        from sklearn.svm import SVR

        """
        RFE works by searching for a subset of features by starting with all features in the training dataset
        and successfully removing features until the desired number remains.

        This is achieved by fitting the given machine learning algorithm used in the core of the model,
        ranking features by importance, discarding the least important features, and re-fitting the model.
        This process is repeated until a specified number of features remains.
        """

        X, y = X, y
        estimator = SVR(kernel="linear")

        # if use RFECV, the number of features to select will be chosen automatically
        rfecv = RFECV(estimator, step=1, cv=5)
        rfecv = rfecv.decision_function(X)
        return rfecv

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
        return sfs

    def removing_low_var(self, df):

        """
        One of the possible ways to reduce dimensionality is to reduce the number of features
        whose variance doesn't meet some threshold.

        :param df: dataframe that consists of words represented as vectors
        """
        sel = VarianceThreshold(threshold=0.0009)
        sel.fit_transform(df)
        return df[df.columns[sel.get_support(indices=True)]]

    def univariate_selection(self, X, words):
        """
        Univariate feature selection works by selecting the best features based on univariate statistical tests.

        :param data: best fit preprocessed data
        """

        X, y = X, words

        X_new = SelectKBest(chi2, k=4)
        fit = X_new.fit(y, X)
        return fit


if __name__ == "__main__":

    select = FeatureSelection()
    vec = Vectorization()
    distance = DistanceMetrics()

    test_text = PreProcessing.txt_preprocess(file_link=
                                             'C:\/Users\Maryna Boroda\Documents\GitHub\DimReduction\exampl_text\wikitext1.txt')
    X, words = vec.vec_TF_IDF(cleaned_words=test_text)
    df = pd.DataFrame(X, columns=[words])

    print(select.tree_based(X, words))

