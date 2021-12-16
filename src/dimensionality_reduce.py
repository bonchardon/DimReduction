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

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import DistanceMetric
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm, datasets
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Dense, Reshape

from scipy import stats
import seaborn as sns

from DimReduction.src.pre_stage import PreProcessing, Vectorization
from DimReduction.src.distance_metrics import DistanceMetrics

pre = PreProcessing()
vec = Vectorization()


class DimensionRed:

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


class Features:

    def __init__(self):
        pass

    """
    Dimensionality reduction for text analysis:
    
    Dimensionality reduction is feature selection/feature extraction. 
    
    Feature selection ==> reduces the dimensions in an univariate manner, i.e. it removes terms on an individual basis as 
    they currently appear without altering them; using this method, we reduce dimensionality by excluding less useful features.
    
    Feature extraction ==> is multivaritate, combining one or more single terms together to produce higher orthangonal terms that
    contain more information and reduce the feature space; here we transform data into lower dimensions.
    """

    def selection(self, txt):

        """
        Types of feature selection for dim reduction:

            (1) recursive feature elimination;
            (2) genetic feature selection;
            (3) sequential forward selection.

        :param txt:
        :return:
        """

        def feature_elimination(data):

            pass

        def feature_selection(data):

            pass

        def forward_selection(data):

            pass

        feature_elimination(data=txt)
        feature_selection(data=txt)
        forward_selection(data=txt)

    def extraction(self, txt):

        """
        Types of feature extraction for dimensionality reduction:

            (1) AutoEncoders (unsupervised Artificial Neural Network);
            (2) PCA;
            (3) LDA.

        :param txt:
        :return:
        """
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

        autoencoder(data=txt)


if __name__ == "__main__":

    """
    In NLP, the aspect of word vectors (as numerical representation of words in corpora/dataset) and dimensionality reduction
    are some of the most important. 
    """
    pre = PreProcessing()
    vec = Vectorization()

    reducing = DimensionRed()
    
    distance = DistanceMetrics()


    """
    Other ways to reduce dimensionality is:
    
        (1) try to train sparse model (like SGDClassifier with huge L1 penalty (why not L2, since it's about euclidiean distance measurement).
        (1.1.) it might help to transform word-count using TF-IDF before using the data in a linear classifier.
        
        (2) we can also use pre-trained dimensionality reducer, such as word2vec / fastText to extract features from text.     
    """


