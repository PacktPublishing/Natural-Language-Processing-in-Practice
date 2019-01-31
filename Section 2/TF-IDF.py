from math import log
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from nltk import corpus
import numpy

# importing the corpus into data variable
data = corpus.brown

# Using Porter Stemmer
stemmer = PorterStemmer()

# Building the list of stop words
# We will filter the tokens against it
stopwords = set(stopwords.words('english'))
stopwords = stopwords.union(string.punctuation)

# Limiting the number of files we will use
# uncomment the following line to use all of the
# corpus files
# fileIds = data.fileids()
fileids = data.fileids()[:30]

idf_matrix  = []
dictionary = dict()

# total count f words in the corpus
words_count = 0

# total count of document in the corpus
documents_count = len(fileids)

# holds the words before and after filtering : Stemming
filtered = dict()

# to save the total counts of every word per file
frequencies = dict()
for fileid in fileids:
    frequencies[fileid] = dict()

# filtering corpus
for fileid in fileids:
    for word in data.words(fileid):
        # Skipping if it is a stop word
        if word in stopwords:
            continue
        # Before and after filtering
        before_word = word
        if before_word in filtered:
            word = filtered[before_word]
        else:
            # stemming the word
            word = stemmer.stem(word)
            filtered[before_word] = word

        if word in frequencies[fileid]:
            frequencies[fileid][word] += 1
        else:
            frequencies[fileid][word] = 1

        # saving all the words in a dictionary
        if word not in dictionary:
            dictionary[word] = words_count
            words_count += 1

# Calculating TF #
# indexes of non zero values
tf_matrix = []
nonzeros = []
for fileid in fileids:
    tf_vector = [0] * words_count
    nonzeros_vec = []
    for word in frequencies[fileid].keys():
        f = frequencies[fileid][word]
        tf_vector[dictionary[word]] = f
        if f > 0:
            nonzeros_vec.append(dictionary[word])
    nonzeros.append(nonzeros_vec)
    tf_matrix.append(tf_vector)

# Calculating IDF
idf_matrix = [0] * words_count
for fileid in fileids:
    for word in frequencies[fileid].keys():
        idf_matrix[dictionary[word]] += 1

# Calculating TF-IDF matrix#
tfidf = []

for i in range(documents_count):
    vector = [0] * words_count
    for j in nonzeros[i]:
        tf_value = tf_matrix[i][j]
        idf_value = idf_matrix[j]
        tf_value = 1 + log(tf_value, 2)
        idf_value = log(1 + documents_count/ float(idf_value), 2)
        vector[j] = tf_value * idf_value
    tfidf.append(vector)

print("------ Top 10 : Keywords per document -------")
for i in range(len(tfidf)):
    print("--- Document  : " + str(fileids[i]))
    vector = tfidf[i]
    sorted = numpy.argsort(vector)[::-1]
    for ind in sorted[:15]:
        stem = list(dictionary.keys())[list(dictionary.values()).index(ind)]
        beforeStemming = list(filtered.keys())[list(filtered.values()).index(stem)]
        print(beforeStemming  + "  -- " + str(vector[ind]))
