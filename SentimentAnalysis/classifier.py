import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
nltk.download('punkt')

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode
# from statistics import multimode
from scipy import stats as s


from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return s.mode(votes)[0]

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(s.mode(votes)[0])
        conf = choice_votes / len(votes)
        return conf

shortPostives = open("positive.txt", encoding='latin-1').read()
shortNegatives = open("negative.txt", encoding='latin-1').read()



documents = []

for r in shortPostives.split('\n'):
  documents.append((r, "pos"))

for r in shortNegatives.split('\n'):
  documents.append((r, "neg"))

all_words = []

short_pos_words = word_tokenize(shortPostives)
short_neg_words = word_tokenize(shortNegatives)

for w in short_pos_words:
  all_words.append(w.lower())

for w in short_neg_words:
  all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

def find_features(document):
  words = word_tokenize(document) #tokenize all words in documents
  features = {}
  for w in word_features:
    features[w] = (w in words) #mark features[w] = 1, if word found in words and 0 otherwise.
  return features

setOfFeatures = [(find_features(rev), category) for(rev, category) in documents]

random.shuffle(setOfFeatures) #shuffle the words to make training efficient.

trainSet = setOfFeatures[:10000]
testSet = setOfFeatures[10000:]



classifier = nltk.NaiveBayesClassifier.train(trainSet)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testSet))*100)
classifier.show_most_informative_features(50)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(trainSet)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testSet))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(trainSet)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testSet))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(trainSet)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testSet))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(trainSet)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testSet))*100)

##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(trainSet)
##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testSet))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(trainSet)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testSet))*100)

# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(trainSet)
# print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testSet))*100)


voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testSet))*100)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

print(sentiment("This movie was wonderful!"))
print(sentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))
