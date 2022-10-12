import gensim
import re
import spacy
import pandas as pd

"""
f = open('train.md', 'r')
lines = f.read()
lines = lines.split('\n')
lines = [re.sub('##.*', '', sent) for sent in lines]
lines = [re.sub('- ', '', sent) for sent in lines]
lines = [re.sub('\(.*\)', '', sent) for sent in lines]
lines = [re.sub('\[|\]', '', sent) for sent in lines]
"""
lines = list(pd.read_csv('questions_and_answers.csv')['QUESTION'])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        
data_words = list(sent_to_words(lines))

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    
#Remove stop words
stop_words = list(nlp.Defaults.stop_words)
data_words = [[word for word in doc if word not in stop_words] for doc in data_words]

dictionary = gensim.corpora.Dictionary(data_words)

lista_words = list(dictionary.itervalues())

with open('context.txt', 'w') as f:
    for item in lista_words:
        f.write("%s\n" % item)