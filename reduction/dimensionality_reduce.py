"""
The overall workflow on dimensionality reduction and distance metrics techniques applied on corpora data collected.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('stopwords')
# nltk.download('wordnet')

from genetic_selection import GeneticSelectionCV
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
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Dense, Reshape

from DimReduction.src.pre_stage import PreProcessing, Vectorization
from DimReduction.src.distance_metrics import DistanceMetrics


"""
    Dimensionality reduction for text analysis:

    Dimensionality reduction is feature selection/feature extraction. 

    Feature selection ==> reduces the dimensions in an univariate manner, i.e. it removes terms on an individual basis as 
    they currently appear without altering them; using this method, we reduce dimensionality by excluding less useful features.

    Feature extraction ==> is multivaritate, combining one or more single terms together to produce higher orthangonal terms that
    contain more information and reduce the feature space; here we transform data into lower dimensions.
"""


class FeatureExtraction:
    """
        Types of feature extraction for dimensionality reduction:

            (1) AutoEncoders (unsupervised Artificial Neural Network);
            (2) PCA;
            (3) LDA.
     """

    def __init__(self):
        pass

    def svm(self, X, words):

        """
        On Support Vector Machines used for dimensionality reduction in low dimensionality vector.

        :param X:
        :param words:
        """

        trans = svm.SVC()
        trans.fit(X)

    def svd(self, X, words):
        n_components = 3
        svd = TruncatedSVD(n_components)
        X_trans_svd = svd.fit_transform(X)
        X_trans_df = pd.DataFrame(X_trans_svd, index=words)
        X_trans_df = X_trans_df.drop_duplicates(X_trans_df.index.duplicated(keep='first'))

        words_cleaned = [w for w in X_trans_df.index]
        return X_trans_df

    def pca(self, X, words):
        # from sparce to dence matrix
        X_dense = X.todense()

        transformer = PCA(n_components=3)
        X_trans_psa = transformer.fit_transform(X_dense)
        X_trans_df_pca = pd.DataFrame(X_trans_psa, index=words)

        # it appears that the number of duplicates is large
        X_trans_df_pca = X_trans_df_pca.drop_duplicates(X_trans_df_pca.index.duplicated(keep='first'))
        words_cleaned = [w for w in X_trans_df_pca.index]
        return X_trans_df_pca

    def factor(self, X, words):

        factor = FactorAnalysis(n_components=3, random_state=0)
        X_factor = factor.fit_transform(X.toarray())

        X_factor_df = pd.DataFrame(X_factor, index=words)
        X_factor_df = X_factor_df.drop_duplicates(X_factor_df.index.duplicated(keep='first'))
        words_cleaned_factor = [w for w in X_factor_df.index]

        print(X_factor_df)

    def isomap(self, X, words):
        trans = Isomap(n_components=10, n_neighbors=10)
        X_trans_isomap = trans.fit_transform(X)
        X_df_isomap = pd.DataFrame(X_trans_isomap)

        X_trans_df = X_df_isomap.drop_duplicates(X_df_isomap.index.duplicated(keep='first'))
        words_cleaned = [w for w in X_trans_df.index]
        print(X_df_isomap)

    def lda(self, X, words):
        latent_dirichlet_trans = LatentDirichletAllocation(n_components=3, max_iter=5, random_state=0)
        X_dirichlet = latent_dirichlet_trans.fit_transform(X)
        X_df_lda = pd.DataFrame(X_dirichlet, index=words)

        X_df_lda = X_df_lda.drop_duplicates(X_df_lda.index.duplicated(keep='first'))
        words_cleaned_lda = [w for w in X_df_lda.index]

    def autoencoder(data):
        encoded_dim = 2
        encoder = Sequential([
            Conv2D(input_shape=(44, 44, 3), filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(encoded_dim)
        ])

        pretrain_encodings = encoder(data).numpy()

        decoder = Sequential([
            Dense(1936, activation='relu', input_shape=(encoded_dim,)),
            Reshape((11, 11, 16)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(3, (3, 3), padding='same')
        ])

        # TODO: start to compile and train autoencoder


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


class Alternative:

    """
    Other ways to reduce dimensionality is:

        1) try to train sparse model (like SGDClassifier with huge L1 penalty.
        (1.1.) it might help to transform word-count using TF-IDF before using the data in a linear classifier.

        (2) we can also use pre-trained dimensionality reducer, such as word2vec/fastText to extract features from text.
    """

    def __init__(self):
        pass


if __name__ == "__main__":

    pre = PreProcessing()
    vec = Vectorization()

    extraction = FeatureExtraction()
    selection = FeatureSelection()
    distance = DistanceMetrics()

    test_text = pre.txt_preprocess(file_link='wikitext1.txt')
    test_text = vec.vec_hash(test_text)

    print(selection.seq_feature_selection(data=test_text))
