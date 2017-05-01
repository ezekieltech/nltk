#stopWords
#a pre-processing steps


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "This is a sample sentence, showing off the stop words filtration."

#stop words are the pre-defined stop words in english
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)


#use the for loop above or use a list comprehension
filtered_sentence = [w for w in word_tokens if not w in stop_words]
##
##filtered_sentence = []
##
##for w in word_tokens:
##    if w not in stop_words:
##        filtered_sentence.append(w)

print(word_tokens)
print('#######################')
print(filtered_sentence)
