##import nltk
##nltk.download('panlex_lite')


from nltk.tokenize import sent_tokenize, word_tokenize

EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

#print(sent_tokenize(EXAMPLE_TEXT))

for word in word_tokenize(EXAMPLE_TEXT):
    print(word)
