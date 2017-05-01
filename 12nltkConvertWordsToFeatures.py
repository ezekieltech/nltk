#pre-paring data for Naives

import nltk
import random
from nltk.corpus import movie_reviews

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
print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))


##do this for all of our documents,
##saving the feature existence booleans
##and their respective positive or negative categories
featuresets = [(find_features(rev), category) for (rev, category) in documents]

