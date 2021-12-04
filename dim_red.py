import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import euclidean_distances
# nltk.download('stopwords')
# nltk.download('wordnet')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap


from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import LatentDirichletAllocation

from scipy import stats
import seaborn as sns


def svd_analysis(file_path):

    """
    In practice TruncatedSVD is useful on large sparse datasets which cannot be centered without making the memory usage explode.

    :param file_path: path to the file that is analyzed
    """

    from scipy.spatial import distance

    # df = pd.read_csv(file_path).head(1000)
    # df = df.drop(['Count'], axis=1)
    # words = df['Words'].values
    # features = df['Log'].values
    # # features = features.reshape((features.shape[0], 1))
    # df_to_analyze = pd.DataFrame([dict(zip(words, features))])

    words = open(file_path, 'r', encoding='utf-8-sig')
    words = words.read().split(' ')

    lemmatizer = WordNetLemmatizer()

    """
    deleting stop words to upgrade overall results
    """

    stop_words = stopwords.words()
    stop_words.extend(['th', 'asd'])
    words = [word for word in words if not word in stop_words]

    """
    decided to use lemmatization as well, to ensure the dimensionality is efficiently reduced
    """
    lemmatized_words = [lemmatizer.lemmatize(w) for w in words]

    vectorizer = CountVectorizer(binary=False, min_df=2)
    X = vectorizer.fit_transform(lemmatized_words)
    feature_names = vectorizer.get_feature_names()
    X = X.toarray()

    perform_pca = True
    if perform_pca:
        n_components = 2
        svd = TruncatedSVD(n_components)
        X_trans = svd.fit_transform(X)
        X_trans_df = pd.DataFrame(X_trans, index=words)
        print(X_trans)

        return svd, vectorizer, feature_names


def factor_analys(file_path):

    words = open(file_path, 'r', encoding='utf-8-sig')
    words = words.read().split(' ')

    vectorizer = CountVectorizer(binary=False, min_df=2)
    X = vectorizer.fit_transform(words)
    X = X.toarray()

    feature_names = vectorizer.get_feature_names()

    transformer = FactorAnalysis(n_components=10, random_state=0)
    X_trans = transformer.fit_transform(X)
    X_trans = pd.DataFrame([x for x in X_trans], index=words)

    '''
    TODO: deal with transformer.explained variance ratio in factor analysis 
    '''

    PC_values = np.arange(transformer.n_components) + 1

    plt.plot(PC_values, transformer, 'ro-', linewidth=2)
    plt.title('Scree Plot SVD')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.show()

    return transformer, vectorizer, feature_names


def pca_analysis(file_path):

    words = open(file_path, 'r', encoding='utf-8-sig')
    words = words.read().split(' ')

    vectorizer = CountVectorizer(binary=False, min_df=2)
    X = vectorizer.fit_transform(words)
    X = X.toarray()

    feature_names = vectorizer.get_feature_names()

    transformer = PCA(n_components=2)
    X_trans = transformer.fit_transform(X)
    X_trans_df = pd.DataFrame(X_trans, index=words)

    return X_trans_df

    # return transformer, vectorizer, feature_names


def isomap_embed(file_path):

    """
    Isomap creates an embedding of teh dataset and attempts tp preserve the relationships in the dataset.
    Non-linear dimensionality reduction method.
    """
    words = open(file_path, 'r', encoding='utf-8-sig')
    words = words.read().split(' ')

    vectorizer = CountVectorizer(binary=False, min_df=2)
    X = vectorizer.fit_transform(words)
    # print(X.shape)

    feature_names = vectorizer.get_feature_names()

    embed = Isomap(n_components=5, n_neighbors=10)
    X_trans = embed.fit_transform(X)
    X_df = pd.DataFrame(X_trans, index=words)

    '''
        TODO: deal with transformer.explained variance ratio in isomap analysis 
    '''
    PC_values = np.arange(embed.n_components) + 1

    plt.plot(PC_values, embed.explained_variance_ratio_, 'ro-', linewidth=2)
    plt.title('Scree Plot SVD')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.show()

    return embed, vectorizer, feature_names


def lda(file_path):

    """
    Linear Discriminant Analysis also works as a dimensionality reduction algorithm,
    it means that it reduces the number of dimension from original to C — 1 number of features where C is the number of classes.

    :param file_path: preprocessed text.
    """

    words = open(file_path, 'r', encoding='utf-8-sig')
    words = words.read().split(' ')

    vectorizer = CountVectorizer(binary=False, min_df=2)
    X = vectorizer.fit_transform(words)

    lda = LinearDiscriminantAnalysis(n_components=10)

    """
    TODO: check TypeError: fit() missing 1 required positional argument: 'y'
    """

    X_trans = lda.fit_transform(X)
    X_trans_df = pd.DataFrame(X_trans, index=words)

    PC_values = np.arange(lda.n_components) + 1

    plt.plot(PC_values, lda.explained_variance_ratio_, 'ro-', linewidth=2)
    plt.title('Scree Plot Linear Discriminant Analysis')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.show()

    return X_trans


def latent_dirichlet(file_path):

    words = open(file_path, 'r', encoding='utf-8-sig')
    words = words.read().split(' ')

    vectorizer = CountVectorizer(binary=False, min_df=2)
    X = vectorizer.fit_transform(words)
    feature_names = vectorizer.get_feature_names()

    lda_ = LatentDirichletAllocation(n_components=10, max_iter=5, random_state=0)
    X_trans = lda_.fit_transform(X)
    X_df = pd.DataFrame(X_trans, index=words)

    """
        TODO: check TypeError: fit() missing 1 required positional argument: 'y'
    """
    PC_values = np.arange(lda_.n_components) + 1

    plt.plot(PC_values, lda_.explained_variance_ratio_, 'ro-', linewidth=2)
    plt.title('Scree Plot Linear Discriminant Analysis')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.show()

    return lda, vectorizer, feature_names


if __name__ == '__main__':

    print(svd_analysis(file_path='wikitext1.txt'))

    # calculate the Euclidean distance between two vectors
    def euclidean_distance(row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return np.sqrt(distance)

    """"
    The way to check and get topics of each of the model
    """

    svd_model, svd_vec, feature_names_svd = svd_analysis(file_path='wikitext1.txt')
    # pca_model, pca_vec, pca_factor_names = pca_analysis(file_path='wikitext1.txt')
    # factor_model, factor_vec, factor_names = factor_analys(file_path='wikitext1.txt')
    # isomap_model, isomap_vec, isomap_factor_names = isomap_embed(file_path='wikitext1.txt')
    # lda_model, lda_vec, lda_feature_names = latent_dirichlet(file_path='wikitext1.txt')

    def display_word_dist(model, feature_names, n_word):
        df = pd.DataFrame()
        for topic_idx, topic in enumerate(model.components_):
            # print("Topic %d:" % (topic_idx))
            words = []
            for i in topic.argsort()[:-n_word - 1:-1]:
                words.append(feature_names[i])
            print(words)
            df['Words'] = words
            print(df)

    display_word_dist(model=svd_model, feature_names=feature_names_svd, n_word=300, components_vecs=svd_components)

