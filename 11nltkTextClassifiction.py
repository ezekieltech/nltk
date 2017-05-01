#Data wrangling

import nltk
import random
from nltk.corpus import movie_reviews


##In each category (we have pos or neg),
##take all of the file IDs (each review has its own ID),
##then store the word_tokenized version (a list of words) for the file ID,
##followed by the positive or negative label in one big list.
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


##shuffle our documents.
##This is because we're going to be training and testing.
##otherwise, chances are we'd train on all of the negatives, some positives,
##and then test only against positives.
##We don't want that, so we shuffle the data.
random.shuffle(documents)

print(documents[1])


##we want to collect all words that we find,
##so we can have a massive list of typical words
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower()) #normalize the data

##perform a frequency distribution,
##to then find out the most common words
all_words = nltk.FreqDist(all_words)

print(all_words.most_common(15))

#how many occurences a word has by doing
print(all_words["stupid"])

