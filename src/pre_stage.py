import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


class PreProcessing:

    def __init__(self):
        pass

    def txt_preprocess(self, file_link):
        example_text = open(file_link, 'r',
                            encoding='utf-8-sig').read()
        example_text = example_text.split(' ')

        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words()
        stop_words.extend(['th', 'asd', 'alphabet\/nalabama'])
        words = [word for word in example_text if not word in stop_words]

        lemmatized_words = [lemmatizer.lemmatize(w) for w in words]

        return lemmatized_words


class Vectorization:

    def __init__(self):
        pass

    def vec_count(self, cleaned_words):

        """


        :param cleaned_words:
        :return:
        """

        vectorizer = CountVectorizer(binary=False, min_df=2)
        X = vectorizer.fit_transform(cleaned_words)
        feature_names = vectorizer.get_feature_names()
        # X = X.toarray()

        return X

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
        vectorizer = HashingVectorizer()
        X = vectorizer.fit_transform(cleaned_words)
        return X

    def vec_bow(self, cleaned_words):
        pass


if __name__ == "__main__":

    PreProcessing()
    Vectorization()

