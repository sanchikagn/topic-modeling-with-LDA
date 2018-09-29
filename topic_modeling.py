import pandas as pd
import numpy as np

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS, stem
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from pprint import pprint

# Data set from https://www.kaggle.com/therohk/million-headlines/data
data = pd.read_csv('resources/abcnews-date-text.csv', error_bad_lines=False)
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text
# print(len(documents))
# print(documents[:5])

np.random.seed(2018)


# Lemmatizing
def lemmatize_stemming(text):
    return stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# Stemming
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# Testing preprocessing
# doc_sample = documents[documents['index'] == 4310].values[0][0]
# print('original document: ')
# words = []
# for word in doc_sample.split(' '):
#     words.append(word)
# print(words)
# print('\n\n tokenized and lemmatized document: ')
# print(preprocess(doc_sample))

# Preprocessing data
processed_docs = documents['headline_text'].map(preprocess)
# print(processed_docs[:10])

# Bag of Words
dictionary = gensim.corpora.Dictionary(processed_docs)
# count = 0
# for k, v in dictionary.iteritems():
#     print(k, v)
#     count += 1
#     if count > 10:
#         break

# Fliter tokens that appear in more that 15 documents, not more than 50% of documents, and keep first 100000 tokens
# of them
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

# Dictionary for each document
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
# print(bow_corpus[4310])

# Applying TF-IDF
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
# for doc in corpus_tfidf:
#     pprint(doc)
#     break

# Train corpus with LDA
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary,
                                       passes=2, workers=2)
# for idx, topic in lda_model.print_topics(-1):
#     print('Topic: {} \nWords: {}'.format(idx, topic))

# LDA with TF-IDF
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary,
                                             passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

# Performance evaluation of LDA with Bag of Words
# for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
#     print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

# Performance Evaluation of LDA with TF-IDF
for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

# Testing
unseen_document = 'How a Pentagon deal became an identity crisis for Google'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
