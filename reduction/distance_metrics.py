from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.neighbors import DistanceMetric

import pandas as pd


class DistanceMetrics:

    def __init__(self):
        pass

    """
    Below are distance metrics for real-valued vector space.
    """

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

    def cheb_dist(self, dataframe):

        dist_cheb = DistanceMetric.get_metric('chebyshev')
        values = dist_cheb.pairwise(dataframe)

        X_cheb = pd.DataFrame(values, index=dataframe.index)

        first_100_words = X_cheb.sort_values(by=[0, 1, 2, 3, 4, 5, 6], ascending=False).index[0:100]
        return first_100_words


if __name__ == "__main__":

   DistanceMetrics()
