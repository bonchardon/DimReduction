"""
The file to unify and clean up all the blocks of code previously developed.
+ visualization task to be handled.
"""

from src.pre_stage import PreProcessing
from src.reduction.feature_selection import FeatureSelection
from src.reduction.feature_extraction import FeatureExtraction
from src.reduction.autoencoder import Autoencoder
from src.reduction.alternative_vars import Alternative


class Group:

    # feature extraction
    TRANS_PCA = 'PCA'
    TRANS_SVD = 'SVD'
    TRANS_FACTOR = 'FactorAnalysis'
    TRANS_ISOMAP = 'Isomap'
    TRANS_LDA = 'LatentDirichletAllocation'
    TRANS_MDS = 'MDS'
    TRANS_LLE = 'LocallyLinearEmbedding'
    TRANS_NCA = 'NeighborhoodComponentsAnalysis'

    #feature selection
    TRANS_RFE = 'RFE'
    TRANS_RFECV = 'RFECV'
    TRANS_LOW_VAR = 'VarianceThreshold'



    def __init__(self):
        pass

    @staticmethod
    def reduce_final(dim_red):

        if dim_red == "PCA":



class Visualize:

    def __init__(self):
        pass


if __name__ =="__main__":

    dim_red = Group()
    vis = Visualize()



