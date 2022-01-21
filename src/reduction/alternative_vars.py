import keras.models

import xgboost

from gensim import models
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import LineSentence
import gensim.downloader
import gensim.downloader as api
from gensim.test.utils import get_tmpfile

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt

from scipy import spatial

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('stopwords')
# nltk.download('wordnet')

# from genetic_selection import GeneticSelectionCV
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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, losses

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten, Dense, Reshape

from src.pre_stage import PreProcessing


class Alternative:

    """
    Other ways to reduce dimensionality is:

        1) try to train sparse model (like SGDClassifier with huge L1 penalty.
        (1.1.) it might help to transform word-count using TF-IDF before using the data in a linear classifier.
        (2) we can also use pre-trained dimensionality reducer, such as word2vec/fastText to extract features from text.
    """

    def __init__(self):
        pass

    def word2vec(self, cleaned_data):
        from sklearn.decomposition import PCA

        word2vec = Word2Vec(cleaned_data, min_count=4)
        word2vec.train(cleaned_data, total_examples=1, epochs=1)

        # in case we want to save and use this later
        # word2vec.save('word2vec.model')
        # model = Word2Vec.load("word2vec.model")

        return word2vec

    def word2vec_pre(self):

        model = api.load('word2vec-google-news-300')
        # model.save('word2vec_pre.model')
        vector = model.wv['computer']
        return vector

    def doc2vec(self, data):

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]
        model = Doc2Vec(documents, vector_size=20, window=2, min_count=1, workers=4)
        # here is the way how to check similarity between docs

        test_doc = nltk.word_tokenize("random bullshit to check the effectiveness of the algo duh")

        return model.docvecs.most_similar(positive=[model.infer_vector(test_doc)], topn=5)

    def glove(self, cleaned_data):

        embeddings = {}

        with open('models/glove.6B.50d.txt', 'rb') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings[word] = vector

            # getting closest word by using glove algo
            def find_closest_embeddings(embedding):
                return sorted(embeddings.keys(), key=lambda word:

            spatial.distance.euclidean(embeddings[word], embedding))

    def fastText(self, word_tokenized_corpus):
        """
        FastText algo may be useful, since with it's help one can get a vector representation of a word
        that is not in a corpus
        """

        embedding_size = 5
        window_size = 5
        min_word = 5
        down_sampling = 5

        ft_model = FastText(word_tokenized_corpus,
                            size=embedding_size,
                            window=window_size,
                            min_count=min_word,
                            sample=down_sampling,
                            sg=1,
                            iter=100)


if __name__ == "__main__":

    vars = Alternative()

    words = PreProcessing.txt_preprocess(
            file_link=
            "C:\/Users\Maryna Boroda\Documents\GitHub\DimReduction\exampl_text\wikitext1.txt")

    words = [sent.split() for sent in words]
    print(vars.doc2vec(words))
