from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.neighbors import DistanceMetric

import pandas as pd


class DistanceMetrics:

    # distance metrics used in the research
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
    CHECBYSHEV = 'chebyshev'

    def __init__(self):
        pass

    """
    Below are distance metrics for real-valued vector space.
    """
    @staticmethod
    def dist_metrics(metrics_type,
                     dataframe,
                     ):
        dist = DistanceMetric.get_metric(metrics_type)
        values = dist.pairwise(dataframe)

        df = pd.DataFrame(values, index=dataframe.index)
        df_dist = df.sort_values(by=[1, 2, 3, 4, 5, 6], ascending=False).index[:500]
        return df_dist

    def cosine_dist(self, dataframe):
        cosine_distance = cosine_similarity(dataframe)
        X_cosine_dist = pd.DataFrame(cosine_distance, index=dataframe.index)

        X_cosine_dist = X_cosine_dist[[0, 1, 2, 3, 4, 5, 6]]
        first_100_words_cosine = X_cosine_dist.sort_values(by=[0, 1, 2, 3, 4, 5, 6], ascending=False).index[0:100]
        return first_100_words_cosine


if __name__ == "__main__":

   DistanceMetrics()
