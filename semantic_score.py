import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
import gensim.downloader


class SemanticScore:
    def __init__(self):
        self.GloveModel = gensim.downloader.load('glove-wiki-gigaword-50')
        self.stopwords = stopwords.words('english')

    def __embed(self, document):
        if isinstance(document, str):
            tokens = nltk.tokenize.word_tokenize(document)
        else:
            tokens = document
        tokens = [token for token in tokens if token not in self.stopwords]
        word_vectors = []
        for Token in list(tokens):
            try:
                vector = self.GloveModel[Token.lower()]
                word_vectors.append(vector.tolist())
            except KeyError:
                continue
        return np.mean(word_vectors, axis=0)

    def similarity(self, document1, document2):
        doc_vector1 = self.__embed(document1)
        doc_vector2 = self.__embed(document2)
        numerator = np.dot(doc_vector1, doc_vector2)
        denominator = np.linalg.norm(doc_vector1) * np.linalg.norm(doc_vector2)
        return numerator / denominator




    