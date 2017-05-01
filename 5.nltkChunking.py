import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
#print(train_text[:30])
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            #use 'chunk,or any word you like
            #putthe chunk you want inside curly braces
            #to bring up or use any part of speech tag, use <>
            #RB + any character, except any line+any character: any for of adverb, any repetition of it with *
    
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            #print(chunked) #to print the chunked stuff

            #chunked.draw() #to see the chunked stuff

            
##            Well, what is happening here is our "chunked" variable is an
##            NLTK tree. Each "chunk" and "non chunk" is
##            a "subtree" of the tree. We can reference these by doing
##            something like chunked.subtrees.
##            We can then iterate through these subtrees like so
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)

            chunked.draw()

    except Exception as e:
        print(str(e))

process_content()
