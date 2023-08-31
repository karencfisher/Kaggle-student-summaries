import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm.notebook import tqdm

from sklearn.metrics import mean_squared_error

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import download
import nltk

import gensim
import gensim.downloader
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from semantic_score import SemanticScore


download('stopwords')
stop_words = stopwords.words('english')

def tokenize_doc(document):
        text = document.lower()
        text = re.sub('\[\d+\]', '', text)
        text = re.sub('\n', ' ', text)
        text = re.sub('[^\w\s]', '', text)
        tokens = nltk.tokenize.word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        filtered_tokens = set()
        for tag in tags:
            if tag[0] in stopwords.words('english'):
                continue
            if tag[1] in ['NN', 'NNP', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                          'JJ', 'JJR', 'JJS']:
                filtered_tokens.add(tag[0])
        return filtered_tokens

def tf_idf(documents, max_words=100):
    doc_tokens = [tokenize_doc(doc) for doc in documents]
    dct = Dictionary(doc_tokens)
    corpus = [dct.doc2bow(document) for document in doc_tokens]
    model = TfidfModel(corpus)

    filtered_docs = []
    for index in tqdm(range(len(corpus))):
        vector = (model[corpus[index]])
        vector.sort(key=lambda x: x[1])
        vector = vector[:max_words]
        filtered_doc = [dct[vect[0]] for vect in vector]
        filtered_docs.append(filtered_doc)
    return filtered_docs

def count_words(text, vocab=False):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    if vocab:
        tokens = set(tokens)
    return len(tokens)

def count_sentences(text, word_count=False):
    sentences = sent_tokenize(text)
    if word_count:
        counts = [count_words(sentence) for sentence in sentences]
        return np.mean(counts)
    return len(sentences)

def objectivity(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_dict = sid.polarity_scores(text)
    return sentiment_dict['neu']

def fitline(x, y, deg):
    coef = np.polyfit(x, y, deg=deg)
    x_vals = np.linspace(np.min(x), np.max(x), num=x.shape[0])
    points = np.polyval(coef, x_vals)
    return x_vals, points

def plot_correlations(x, df, deg):
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    corr = np.corrcoef(x, df['content'])
    x_vals, trend = fitline(x, df['content'], deg=deg)

    sns.scatterplot(x=x, y=df['content'], ax=axes[0])
    axes[0].plot(x_vals, trend, color='red')
    axes[0].set_title(f'Correlation = {corr[0, 1]: .3f}')
    axes[0].set_ylabel('Content Score')

    corr = np.corrcoef(x, y=df['wording'])
    x_vals, trend = fitline(x, df['wording'], deg=deg)
    rmse = np.sqrt(mean_squared_error(df['wording'], trend))

    sns.scatterplot(x=x, y=df['wording'], ax=axes[1])
    axes[1].plot(x_vals, trend, color='red')
    axes[1].set_title(f'Correlation = {corr[0, 1]: .3f}')
    axes[1].set_ylabel('Wording Score')
    plt.show()