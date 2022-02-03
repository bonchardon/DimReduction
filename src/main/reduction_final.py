"""
The file to unify and clean up all the blocks of code previously developed.
+ visualization task to be handled.
"""

import plotly.express as px
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.spatial import distance

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.main.pre_stage import PreProcessing
from src.main.reduction.feature_selection import FeatureSelection
from src.main.reduction.feature_extraction import FeatureExtraction
from src.main.reduction.autoencoder import Autoencoder
from src.main.reduction.alternative_vars import Alternative
from src.main.distance_metrics import DistanceMetrics

from sklearn.manifold import TSNE


class DimRed:

    # feature extraction
    TRANS_PCA = 'PCA'
    TRANS_SVD = 'SVD'
    TRANS_FACTOR = 'FactorAnalysis'
    TRANS_ISOMAP = 'Isomap'
    TRANS_LDA = 'LatentDirichletAllocation'
    TRANS_MDS = 'MDS'
    TRANS_LLE = 'LocallyLinearEmbedding'
    TRANS_NCA = 'NeighborhoodComponentsAnalysis'

    # feature selection
    TRANS_RFE = 'RFE'
    TRANS_RFECV = 'RFECV'
    TRANS_LOW_VAR = 'VarianceThreshold'

    def __init__(self):
        pass

    @staticmethod
    def reduce_final(dim_red, X, words,
                     kwargs=None):

        trans = None

        if dim_red == DimRed.TRANS_PCA:
            trans = FeatureExtraction.pca(X, words)

        return trans(**kwargs)


class Visualize:

    def __init__(self, value1, value2):
        self.value1 = value1
        self.value2 = value2

        X = list(value1)
        Y = list(value2)

        fig, ax = plt.subplots()
        ax.scatter(X, Y)

        for i, txt in enumerate(words):
            ax.annotate(txt, (X[i], Y[i]))

        plt.show()


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


if __name__ =="__main__":

    dim_red = DimRed()
    # vis = Visualize()
    FeatureExtraction = FeatureExtraction()

    # words = PreProcessing.txt_preprocess(
    #     file_link=
    #     "C:\/Users\Maryna Boroda\Documents\GitHub\DimReduction\exampl_text\wikitext1.txt")

    words = PreProcessing.txt_preprocess(
        file_link=
        "C:\/Users\Maryna Boroda\Documents\GitHub\DimReduction\exampl_text\/text2.txt")

    words = [sent.split() for sent in words]
    words = [item for sublist in words for item in sublist]
    model = pickle.load(open("C:\/Users\Maryna Boroda\Documents\GitHub\DimReduction\src\main\/reduction\glove_word2vec.sav", 'rb'))

    subset = np.array([model[word] for word in words])

    reduced = pca.compute_pca(subset)
    reduced_aut = FeatureExtraction.pca(X=subset, words=words)
    print(reduced_aut)

    df_pca = pd.DataFrame(reduced, columns='X Y'.split())
    df_pca.index = words
    # print(df_pca)

    """
    Just trying here some vizualization
    """

    tsne = TSNE(n_components=3)
    X_tsne = tsne.fit_transform(subset)
    df_tsne = pd.DataFrame(X_tsne, columns='X Y Z'.split())
    df_tsne.index = words
    print(df_tsne)




    plt.style.use('seaborn')

    X = list(df_tsne['X'])
    Y = list(df_tsne['Y'])
    Z = list(df_tsne['Z'])
    # colors = {'Setosa': '#FCEE0C', 'Versicolor': '#FC8E72', 'Virginica': '#FC3DC9'}

    fig, ax = plt.subplots()
    ax.scatter(X, Y, c=Y,
               # norm=plt.colors.Normalize,
               cmap="nipy_spectral"
               )

    for i, txt in enumerate(words):
        ax.annotate(txt, (X[i], Y[i]))

    plt.title("First three PCA components")
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.xlabel('1st PCA component')
    plt.ylabel('2nd PCA component')
    plt.show()















    # TODO: our dataframe with word2vec vectors
    #  measure closest values using cosine similarity, euclidean, etc.
    from scipy.spatial.distance import cdist, pdist
    from scipy.spatial import distance
    data = {"word": words, "vector": list(subset)}
    df = pd.DataFrame(data)

    print(df)
    #
    # print(distance.cdist(subset, subset, 'euclidean'))
