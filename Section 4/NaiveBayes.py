
# Getting the dataset
from sklearn.datasets import fetch_20newsgroups

# Getting the training and test data subsets
newsgroup_train = fetch_20newsgroups(subset='train', shuffle=True)
newsgroup_test = fetch_20newsgroups(subset='test', shuffle=True)

# Checking out the Categories Names
i = 0
for cat in newsgroup_train.target_names:
    i = i + 1
    print(str(i) + " - " + str(cat))


# Printing a single ost
print("\n".join(newsgroup_train.data[5].split("\n")[:10]))

# Extracting features
from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()
newsgroup_train_counts = count_vector.fit_transform(newsgroup_train.data)

# Calculating TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
newsgroup_train_tfidf = tfidf_transformer.fit_transform(newsgroup_train_counts)


# Training Naive Bayes
from sklearn.naive_bayes import MultinomialNB

nb_cla = MultinomialNB().fit(newsgroup_train_tfidf, newsgroup_train.target)


# Simplifying the process with a Pipeline
from sklearn.pipeline import Pipeline

NB_Classifier = Pipeline([('vectorizer', CountVectorizer()), ('tfidf_matrix', TfidfTransformer()), ('nb_classifier', MultinomialNB())])
NB_Classifier = NB_Classifier.fit(newsgroup_train.data, newsgroup_train.target)

# Testing the Classifier
import numpy as np

predicted = NB_Classifier.predict(newsgroup_test.data)
print(np.mean(predicted == newsgroup_test.target))


