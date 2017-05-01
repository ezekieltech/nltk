#pre-paring data for Naives
import pickle
import nltk
import random
from nltk.corpus import movie_reviews

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)


#we want to use just 3000 words
#new variable, word_features, which contains random 3,000 most common words
word_features = list(all_words.keys())[:3000]

'''
##This is produces top 3000 words
##It is better, and it improves accuracy
word_features = [w[0] for w in sorted(all_words.items(), key=lambda k_v:k_v[1], reverse=True)[:3000]]
'''


##build a quick function that will find these top 3,000 words in our positive
##and negative documents, marking their presence as either positive or negative:

def find_features(document):
    words = set(document) #list of words converts to a set
    features = {}
    for w in word_features:
        features[w] = (w in words) #if w in word_features, bring True, otherwise False

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))


##do this for all of our documents,
##saving the feature existence booleans
##and their respective positive or negative categories
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]


'''
#posterior = prior occurences * likelihood / evidence
##Next, we can define, and train our classifier like:
classifier = nltk.NaiveBayesClassifier.train(training_set)

##Next, we can test it:
#print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)


classifier.show_most_informative_features(15)
'''
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

'''
##Can you imagine having to train the classifier every time you wanted to fire it
##up and use it? What horror! Instead, what we can do is use the Pickle module to
##go ahead and serialize our classifier object, so that all we need to do is load
##that file in real quick.

##This opens up a pickle file, preparing to write in bytes some data.
##Then, we use pickle.dump() to dump the data.
##The first parameter to pickle.dump() is what are you dumping,
##the second parameter is where are you dumping it.
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()



##We open the file to read as bytes.
##Then, we use pickle.load() to load the file,
##and we save the data to the classifier variable.
##Then we close the file, and that is that.
##We now have the same classifier object as before!
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
'''
