import nltk
from nltk.tag import StanfordNERTagger
from nltk.metrics.scores import accuracy

raw_annotations = open("/usr/share/wikigold.conll.txt").read()
split_annotations = raw_annotations.split()

# Amend class annotations to reflect Stanford's NERTagger
for n,i in enumerate(split_annotations):
	if i == "I-PER":
		split_annotations[n] = "PERSON"
	if i == "I-ORG":
		split_annotations[n] = "ORGANIZATION"
	if i == "I-LOC":
		split_annotations[n] = "LOCATION"

# Group NE data into tuples
def group(lst, n):
  for i in range(0, len(lst), n):
	val = lst[i:i+n]
	if len(val) == n:
	  yield tuple(val)

reference_annotations = list(group(split_annotations, 2))

pure_tokens = split_annotations[::2]
##This reads in the data, splits it by spacing, then subsets everything in split_annotations by an increment of 2 (starting with the 0th element). This produces a dataset like the following (much smaller) example:
##
##['Founding', 'member', 'Kojima', 'Minoru', 'played', 'guitar', 'on', 'Good', 'Day', ',', 'and', 'Wardanceis', 'cover', 'of', 'a', 'song', 'by', 'UK', 'post', 'punk', 'industrial', 'band', 'Killing', 'Joke', '.']

tagged_words = nltk.pos_tag(pure_tokens)
nltk_unformatted_prediction = nltk.ne_chunk(tagged_words)

#Convert prediction to multiline string and then to list (includes pos tags)
multiline_string = nltk.chunk.tree2conllstr(nltk_unformatted_prediction)
listed_pos_and_ne = multiline_string.split()

# Delete pos tags and rename
del listed_pos_and_ne[1::3]
listed_ne = listed_pos_and_ne

# Amend class annotations for consistency with reference_annotations
for n,i in enumerate(listed_ne):
	if i == "B-PERSON":
		listed_ne[n] = "PERSON"
	if i == "I-PERSON":
		listed_ne[n] = "PERSON"    
	if i == "B-ORGANIZATION":
		listed_ne[n] = "ORGANIZATION"
	if i == "I-ORGANIZATION":
		listed_ne[n] = "ORGANIZATION"
	if i == "B-LOCATION":
		listed_ne[n] = "LOCATION"
	if i == "I-LOCATION":
		listed_ne[n] = "LOCATION"
	if i == "B-GPE":
		listed_ne[n] = "LOCATION"
	if i == "I-GPE":
		listed_ne[n] = "LOCATION"

# Group prediction into tuples
nltk_formatted_prediction = list(group(listed_ne, 2))
