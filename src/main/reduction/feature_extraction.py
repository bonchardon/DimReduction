
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

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.manifold import TSNE
from sklearn import svm, datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale

from src.main.pre_stage import PreProcessing, Vectorization
from src.main.distance_metrics import DistanceMetrics


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
            (3) LDA, etc.
     """

    def __init__(self):
        pass

    def svd(self, X, words):

        n_components = 5
        svd = TruncatedSVD(n_components, random_state=42)
        X_trans_svd = svd.fit_transform(X)

        # X_trans_df = pd.DataFrame(X_trans_svd, index=words)
        # X_trans_df = X_trans_df.drop_duplicates(X_trans_df.index.duplicated(keep='first'))
        # words_cleaned = [w for w in X_trans_df.index]

        # for i, comp in enumerate(svd.components_):
        #     terms_in_comp = zip(words, comp)
        #     # sorted_terms = sorted(terms_in_comp, key=lambda x: x[1], reverse=True)
        #     sorted_terms = sorted(terms_in_comp, key=lambda x: x[1], reverse=True)[:10]
        #     print("Concept %d: " % i)
        #     # keywords = []
        #     for term in sorted_terms:
        #         print(term[0])
        #     # return keywords
        #     print(" ")

        return svd.explained_variance_ratio_

    def pca(self, X, words):
        # from sparce to dence matrix
        # X_dense = X.todense()

        transformer = PCA(n_components=2)
        X_trans_psa = transformer.fit_transform(X)
        X_trans_df_pca = pd.DataFrame(X_trans_psa, index=words)

        # it appears that the number of duplicates is large
        # X_trans_df_pca = X_trans_df_pca.drop_duplicates(X_trans_df_pca.index.duplicated(keep='first'))
        words_cleaned = [w for w in X_trans_df_pca.index]
        return X_trans_df_pca

    def factor(self, X, words):

        factor = FactorAnalysis(n_components=3, random_state=0)
        X_factor = factor.fit_transform(X.toarray())

        X_factor_df = pd.DataFrame(X_factor, index=words)
        X_factor_df = X_factor_df.drop_duplicates(X_factor_df.index.duplicated(keep='first'))
        words_cleaned_factor = [w for w in X_factor_df.index]

        return X_factor_df

    def isomap(self, X, words):
        trans = Isomap(n_components=10, n_neighbors=10)
        X_trans_isomap = trans.fit_transform(X)

        X_df_isomap = pd.DataFrame(X_trans_isomap)
        X_trans_df = X_df_isomap.drop_duplicates(X_df_isomap.index.duplicated(keep='first'))
        words_cleaned = [w for w in X_trans_df.index]
        return X_df_isomap

    def lda(self, X, words):
        latent_dirichlet_trans = LatentDirichletAllocation(n_components=3, max_iter=5, random_state=0)
        X_dirichlet = latent_dirichlet_trans.fit_transform(X)

        X_df_lda = pd.DataFrame(X_dirichlet, index=words)
        X_df_lda = X_df_lda.drop_duplicates(X_df_lda.index.duplicated(keep='first'))
        words_cleaned_lda = [w for w in X_df_lda.index]
        return X_df_lda

    def mds(self, X, words):

        """
        Multidimensional Scaling (using Scikit-Learn).

        Normally the distance measure used in MDS is the Euclidean distance,
        however, any other suitable dissimilarity metric can be used when applying MDS.

        :param data: vectorized text data to be analyzed.
        """

        # need to reshape an array since 'expected 2D array, got scalar array instead'
        X = float(X)
        # X = X.reshape(1, -1)

        # mds = MDS(n_components=3,
        #               metric=True,
        #               n_init=4,
        #               max_iter=300,
        #               verbose=0,
        #               eps=0.001,
        #               n_jobs=None,
        #               random_state=42,
        #               dissimilarity='euclidean')

        mds = MDS(n_components=3)
        X_trans = mds.fit_transform(X)
        stress = mds.stress_

        X_trans_df = pd.DataFrame(X_trans, index=words)
        return X_trans

    def lle(self, X, words):

        """
        According to scikit-learn documentation:

        "Locally linear embedding (LLE) seeks a lower-dimensional projection of the data
         which preserves distances within local neighborhoods."

        """

        # X = X.toarray()
        lle = LocallyLinearEmbedding(n_components=3, eigen_solver='dense')
        X_trans = lle.fit(X)

        X_df_lle = pd.DataFrame(X_trans, index=words)
        X_trans_df = X_df_lle.drop_duplicates(X_df_lle.index.duplicated(keep='first'))
        return X_trans_df

    def nca(self, X, words):

        """
        Neighborhood Components Analysis (NCA) tries to find a feature space
        such that a stochastic nearest neighbor algorithm will give the best accuracy.
        Like LDA, it is a supervised method.


        :param X:
        :param words:
        """

        nca = NeighborhoodComponentsAnalysis(n_components=3)

        pass

    def t_SNE(self, df):
        tsne_try = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(subset)
        df_tsne = pd.DataFrame(tsne_try, columns='X Y'.split())
        df_tsne.index = words_try
        print(df_tsne)


if __name__ == "__main__":

    vec = Vectorization()
    extraction = FeatureExtraction()
    distance = DistanceMetrics()

    test_text = PreProcessing.txt_preprocess(file_link=
                                   'C:\/Users\Maryna Boroda\Documents\GitHub\DimReduction\exampl_text\wikitext1.txt')
    X, words = vec.vec_TF_IDF(cleaned_words=test_text)

    # print(X, words)

    df1 = pd.DataFrame(
                    X,
                    index=words,
                    columns=["TF-IDF"]
                        ).sort_values("TF-IDF", ascending=False)

    print(extraction.svd(X, words))
