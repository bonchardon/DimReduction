import nltk
import logging
import pandas as pd
from nltk.corpus import stopwords
stopwords.words('english')
from nltk.stem import WordNetLemmatizer

from itertools import islice

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import make_pipeline
from eli5.sklearn import InvertableHashingVectorizer
from sklearn.linear_model import SGDClassifier


class PreProcessing:

    def __init__(self):
        pass

    @staticmethod
    def txt_preprocess(file_link):
        example_text = open(file_link, 'r',
                            encoding='utf-8-sig').readlines()
        # example_text = example_text.split(' ')

        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words('english')
        stop_words.extend(['th', 'asd', 'alphabet\/nalabama',
                            'the', 'of', 'and'])

        wrds = [text.split() for text in example_text]
        wrds = [[w for w in text if not w in stop_words] for text in wrds]

        lems = [
            [lemmatizer.lemmatize(w) for w in lst] for lst in wrds
        ]
        lemmatized_texts = [' '.join(x) for x in lems]
        return lemmatized_texts


class Vectorization:

    def __init__(self):
        pass

    def vec_count(self, cleaned_words):

        """
        Applying Bag of Words algo. BOW algo can be implemented in a number of ways (i.e., with libraries Reras, Gensim, etc.).

        :param cleaned_words: preprocessed and cleaned dataset that is separated in chunks.
        """

        vectorizer = CountVectorizer(binary=False, min_df=2)
        X = vectorizer.fit_transform(cleaned_words)
        X = np.log(X.toarray() + 1)
        feature_names = vectorizer.get_feature_names()
        # df = pd.DataFrame(X.toarray().transpose(), index=vectorizer.get_feature_names())

        return X, feature_names

    def vec_TF_IDF(self, cleaned_words):

        """
        Applying Tf-Idf vectorizer for features extraction and getting outputs as scientific notation.
        Note an issue: all the values, vectors are the same, so check this issue.

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

        vectorizer = TfidfVectorizer()

        X = vectorizer.fit_transform(cleaned_words)
        X = X.toarray()

        # pd.set_option('display.float_format', lambda x: '%.e2' % x)
        # df = pd.DataFrame(
        #                     X,
        #                     columns=vectorizer.get_feature_names()
        #                     )
        # dataframe where we can see all the most 'valuable' tokens
        # df1 = pd.DataFrame(
        #                     X,
        #                     index=vectorizer.get_feature_names(),
        #                     columns=["TF-IDF"]
        # ).sort_values("TF-IDF", ascending=False)
        return X, vectorizer.get_feature_names()

    def vec_hash(self, cleaned_words):

        """
        One of the ways from debugging HashingVectorizer:

        Note: "There is no way to compute the inverse transform (from feature indices to string feature names)
        which can be a problem when trying to introspect which features are most important to a model."
        """
        vectorizer = HashingVectorizer(stop_words='english', ngram_range=(1,2))
        X = vectorizer.fit_transform(cleaned_words)

        # ivec = InvertableHashingVectorizer(vectorizer)
        # ivec.fit(cleaned_words)

        # df = pd.DataFrame(
        #     X.todense(),
        #     columns=ivec.get_feature_names()
        # )

        return vectorizer.get_stop_words()


class Scaling:

    def __init__(self):
        pass

    def normalization(self):

        """
        Normalization is a rescaling of the data from the original range
        so that all values are within the new range of 0 and 1.

        A value is normalized as follows:

                y = (x – min) / (max – min)

        """
        pass

    def standardization(self):

        """
        Standardizing a dataset involves rescaling the distribution of values
        so that the mean of observed values is 0 and the standard deviation is 1.

        A value is standardized as follows:

                y = (x – mean) / standard_deviation


        Where the mean is calculated as:

                mean = sum(x) / count(x)
        And the standard_deviation is calculated as:

                standard_deviation = sqrt( sum( (x – mean)^2 ) / count(x))
        """
        pass


class MultoColl:

    def __init__(self):
        pass

    """
    Checking for multicollinearity is a very important step during the feature selection process. 
    Multicollinearity can significantly reduce the model’s performance. 
    """

    @staticmethod
    def vif(df):

        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        """
        VIF is a number that determines whether a variable has multicollinearity or not.
        That number also represents how much a variable is inflated because of the linear dependence with other variables.
        
        :param df: preprocessed and cleaned data, ready to be used in the feature selection analysis.
        """
        # TODO: deal with inf problem (all the results are inf).
        #   possible issue: This shows a perfect correlation between two independent variables.
        #   In the case of perfect correlation, we get R2 =1, which lead to 1/(1-R2) infinity.

        vif_info = pd.DataFrame()
        vif_info['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        vif_info['Column'] = df.columns
        return vif_info.sort_values('VIF', ascending=False)


if __name__ == "__main__":

    logging.basicConfig()
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    pre = PreProcessing()
    vec = Vectorization()

    test_text = PreProcessing.txt_preprocess(file_link="C:\/Users\Maryna Boroda\Documents\GitHub\DimReduction\exampl_text\wikitext1.txt")
    X, words = vec.vec_count(cleaned_words=test_text)

    df = pd.DataFrame(X, columns=[words])
    print(df.reset_index(drop=True))
    print(MultoColl.vif(df=df))
