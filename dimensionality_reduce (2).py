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

from scipy import stats
import seaborn as sns


class PreProcessing:

    def __init__(self):
        pass

    def txt_preprocess(self, file_link):
        example_text = open(file_link, 'r',
                            encoding='utf-8-sig').read()
        example_text = example_text.split(' ')
        print(len(example_text))

        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words()
        stop_words.extend(['th', 'asd', 'alphabet\/nalabama'])
        words = [word for word in example_text if not word in stop_words]

        lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
        print(len(lemmatized_words))

        return lemmatized_words

    def vec_count(self, cleaned_words):

        """
        CountVectorizer helps us to understand the type of a text by the frequency of words in it.
        Thus, it might be compared to count distribution.

        Using vec.fit_transform we can get count of the input data.
        """

        vectorizer = CountVectorizer(binary=False, min_df=2)
        X = vectorizer.fit_transform(cleaned_words)
        feature_names = vectorizer.get_feature_names()
        # X = X.toarray()

        return X

    def vec_TF_IDF(self, cleaned_words):

        """
        Applying Tf-Idf vectorizer for features extraction and getting outputs as scientific notation.

        :param cleaned_words: preprocessed and cleaned data

        TfIdfVectorizer parameters used:
        :param ngram_range:     The lower and upper boundary of the range of n-values for different n-grams to be extracted.
                                All values of n such that min_n <= n <= max_n will be used.
                                For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams.
        :param max_df:          When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
                                If float in range [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts.
        :param min_df:          When building the vocabulary ignore terms that have a document frequency strictly lower
                                than the given threshold.
                                This value is also called cut-off in the literature.
                                If float in range of [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts.
                                This parameter is ignored if vocabulary is not None.
        """

        vectorizer = TfidfVectorizer(min_df=3, max_df=0.5, ngram_range=(1, 2))
        X = vectorizer.fit_transform(cleaned_words)

        #
        pd.set_option('display.float_format', lambda x: '%.e2' % x)
        df = pd.DataFrame(
                            X.todense(),
                            columns=vectorizer.get_feature_names()
                            )

        return df

    def vec_hash(self, cleaned_words):

        """
        Convert a collection of text documents to a matrix of token occurrences.


        """
        vectorizer = HashingVectorizer()
        X = vectorizer.fit_transform(cleaned_words)
        return X

    def word2vec(self, cleaned_words):

        """
        Important note: the length of the corpus (for training) must be quite lengthy and of good quality.

        """
        pass

    def word2vec_tfidf(self, cleaned_words):

        pass


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
    they currently appear without altering them, 
    Feature extraction ==> is multivaritate, combining one or more single terms together to produce higher orthangonal terms that
    contain more information and reduce the feature space.
    """

    def selection(self, txt):

        pass

    def extraction(self, txt):

        pass


class DistanceMetrics:

    def __init__(self):
        pass

    def euclid_dist(self, dataframe):

        # one of the possible alternatives
        distance_matrix = euclidean_distances(dataframe)

        dist = DistanceMetric.get_metric('euclidean')
        values = dist.pairwise(dataframe)

        X_euclid = pd.DataFrame(values, index=dataframe.index)

        first_100_words = X_euclid.sort_values(by=[0, 1, 2, 3, 4, 5, 6], ascending=False).index[0:100]
        return first_100_words

    def cosine_dist(self, dataframe):
        cosine_distance = cosine_similarity(dataframe)
        X_cosine_dist = pd.DataFrame(cosine_distance, index=dataframe.index)

        X_cosine_dist = X_cosine_dist[[0, 1, 2, 3, 4, 5, 6]]
        first_100_words_cosine = X_cosine_dist.sort_values(by=[0, 1, 2, 3, 4, 5, 6], ascending=False).index[0:100]
        return first_100_words_cosine

    def manhattan_dist(self, dataframe):
        dist_manhattan = DistanceMetric.get_metric('manhattan')
        values_manhattan = dist_manhattan.pairwise(dataframe)

        X_manhattan = pd.DataFrame(values_manhattan, index=dataframe.index)

        X_manhat_dist = X_manhattan[[0, 1, 2, 3, 4, 5, 6]]

        first_100_words_manhattan = X_manhat_dist.sort_values(by=[0, 1, 2, 3, 4, 5, 6],
                                                              ascending=False).index[0:100]
        return first_100_words_manhattan


if __name__ == "__main__":

    preprocessing = PreProcessing()
    reducing = DimensionRed()
    distance = DistanceMetrics()

    print(preprocessing.vec_TF_IDF(cleaned_words=preprocessing.txt_preprocess(file_link='wikitext1.txt')))



