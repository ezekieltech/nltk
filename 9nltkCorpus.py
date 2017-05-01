###where is the nltk package in my disk? use this
##import nltk
##print(nltk.__file__)


##in the file, you will see this:
##        os.path.join(sys.prefix, str('lib'), str('nltk_data')),
##        os.path.join(
##            os.environ.get(str('APPDATA'), str('C:\\')), str('nltk_data'))
#the corpus data is in appdata folde. which can be accessed with %appdata% on the command line


#here is you can access the data
#you can access via python, like accessing a folder but this may be better
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg

# sample text
sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

for x in range(5):
    print(tok[x])

#print(tok[5:15])

