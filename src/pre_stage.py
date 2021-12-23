import nltk
import logging
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from itertools import islice

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import make_pipeline
from eli5.sklearn import InvertableHashingVectorizer
from sklearn.linear_model import SGDClassifier


class PreProcessing:

    def __init__(self):
        pass

    @classmethod
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

    def chunks(self, data, size=1000):
        it = iter(data)
        for i in range(0, len(data), size):
            yield list(data[k] for k in islice(it, size))


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
        feature_names = vectorizer.get_feature_names()
        X = X.toarray()

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

        chunk1 = ' '.join(cleaned_words[0:3000])
        chunk2 = ' '.join(cleaned_words[3000:6000])
        chunk3 = ' '.join(cleaned_words[6000:8000])
        chunk4 = ' '.join(cleaned_words[8000:9000])
        chunk5 = ' '.join(cleaned_words[9000:])

        chunks = [chunk1, chunk2, chunk3, chunk4, chunk5]

        vectorizer = TfidfVectorizer()

        X = vectorizer.fit_transform(chunks)
        matrix = X.todense()
        list_dense = matrix.tolist()

        #
        pd.set_option('display.float_format', lambda x: '%.e2' % x)
        df = pd.DataFrame(
                            list_dense,
                            columns=vectorizer.get_feature_names()
                            )
        # dataframe where we can
        df1 = pd.DataFrame(
                            X[0].T.todense(),
                            index=vectorizer.get_feature_names(),
                            columns=["TF-IDF"]
        ).sort_values("TF-IDF", ascending=False)

        return df1

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


if __name__ == "__main__":

    logging.basicConfig()
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    pre = PreProcessing()
    test_text = pre.txt_preprocess(file_link='wikitext1.txt')

    vec = Vectorization()
    test_text_try = vec.vec_TF_IDF(cleaned_words=test_text)

    print(test_text_try)

