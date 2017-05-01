#look up similar and opp and context of words

from nltk.corpus import wordnet

###sysnonyms of the word program
####products a list which can be refrenced directly
##syns = wordnet.synsets("program")
##
###printsynset
##print (syns[0]) #reference the 1st element
##
###just the word
##print(syns[0].lemmas()[0].name())
##
###definition
##print(syns[0].definition())
##
###examples like in context
##print(syns[0].examples())

synonyms = []
antonyms = []
for syn in wordnet.synsets('good'):
    for l in syn.lemmas():
        #print('l: ',l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

##print(set(synonyms))
##print(set(antonyms))


#semantic similarities
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cat.n.01')
print(w1.wup_similarity(w2))
