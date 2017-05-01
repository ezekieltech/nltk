### -*- coding: utf-8 -*-
##
##from nltk.tag import StanfordNERTagger
##from nltk.tokenize import word_tokenize
##
##from nltk.internals import find_jars_within_path
###parser = StanfordParser(model_path="path/to/englishPCFG.ser.gz")
###parser._classpath = tuple(find_jars_within_path(stanford_dir))
##
##

##
##st = StanfordNERTagger(path_to_model,path_to_jar, encoding='utf-8')
##st._clsspath = tuple(find_jars_within_path(stanford_dir))
##
##text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'
##
##tokenized_text = word_tokenize(text)
##classified_text = st.tag(tokenized_text)
##
##print(classified_text)

##import os
##if os.environ.get("JAVA_HOME") is not None and "/bin" not in os.environ["JAVA_HOME"]:
##    os.environ["JAVAHOME"] = os.path.normpath(os.path.join(os.environ["JAVA_HOME"], "bin"))

##import os
##
##os.environ['JAVA_HOME'] = "C:\\Program Files\\Java\\jdk1.8.0_121\\bin"


# -*- coding: utf-8 -*-
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

nltk.internals.config_java("C:\Program Files\Java\jdk1.8.0_121\\bin\java.exe")
path_to_jar = 'C:\\dataScience\\stanford-ner-2016-10-31\\stanford-ner-3.7.0.jar'
path_to_model = 'C:\\dataScience\\stanford-ner-2016-10-31\\classifiers\\english.all.3class.distsim.crf.ser.gz'
st = StanfordNERTagger(path_to_model, path_to_jar, encoding='utf-8')

text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'

tokenized_text = word_tokenize(text)
classified_text = st.tag(tokenized_text)

print(classified_text)
