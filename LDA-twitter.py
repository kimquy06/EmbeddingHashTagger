# Picle
import pickle
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


# Plotting
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

#Path to data
path_data="./../../EmTagger/"
data_name="all_preproc_no_hashtags.pickle"


with open(path_data+data_name, 'rb') as f:
    d = pickle.load(f, encoding='latin1')

#print(d[1:4])

data = [i['text'].split() for i in d]

length_data=100000

#print(data[1:15])
# Create Dictionary
id2word = corpora.Dictionary(data)

# Create Corpus
texts = data[0:length_data]

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
#print(corpus[:1])

# Human readable
#print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20,
                                           random_state=100,
                                     	   n_jobs = -1, 
                                           chunksize=256,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

pprint(lda_model.print_topics(5))
path_saved_model = './tmp/model_lda.atmodel'
lda_model.save(path_saved_model)
#model = gensim.models.ldamodel.LdaModel.load(path_saved_model)

doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
Visualize
