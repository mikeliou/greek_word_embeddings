'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
import string
import time
import json
import re
from tqdm import tqdm
from operator import itemgetter
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

class EpochLogger(CallbackAny2Vec):
	'''Callback to log information about training'''

	def __init__(self):
		self.epoch = 0

	def on_epoch_begin(self, model):
		print("Epoch #{} start".format(self.epoch))

	def on_epoch_end(self, model):
		print("Epoch #{} end".format(self.epoch))
		self.epoch += 1

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='/home/mlc/Desktop/gow/datasets/testData.txt',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='gow.vec',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=100,
	                    help='Number of dimensions. Default is 100.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', type=int, default=1,
                        help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	return parser.parse_args()

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	#walks = list([map(str, walk) for walk in walks])
	epoch_logger = EpochLogger()
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter, callbacks=[epoch_logger])
	model.wv.save_word2vec_format(args.output)
	
	return

def load_file(filename):
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            docs.append(line)

    return docs

def clean_str(s):
    s = re.sub(r"[^A-Za-zΑ-Ωα-ωΆ-Ώά-ώ0-9(),!?\'\`]", " ", s)     
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.lower()

    return s.strip().split()

def preprocessing(docs):
	preprocessed_docs = []
	gr_stops = list(stopwords.words('greek'))
	gr_stops_ext = ['της', 'τη', 'τους']
	for word_ext in gr_stops_ext:
		if word_ext not in gr_stops:
			gr_stops.append(word_ext)

	for doc in docs:
		cl_str = clean_str(doc)
		#for cl_word in cl_str:
		#	if cl_word in gr_stops:
		#		cl_str.remove(cl_word)
		cl_str = [word for word in cl_str if word not in gr_stops]
		preprocessed_docs.append(cl_str)

	return preprocessed_docs

def build_words_dict(docs):
    words_dict = dict()
    word_id = 0

    for doc in docs:
        for word in doc:
            if word not in words_dict:
                word_id = word_id + 1
                words_dict[word] = word_id
        
    return words_dict

def create_graphs_of_words(docs, window_size):
    """ 
    Create graphs of words
    """
    G = nx.Graph()
    #max_docs = len(docs)
    #doc_count = 0
    for doc in tqdm(docs):
        #doc_count = doc_count + 1
        if len(doc) == 0: continue
        for i in range(len(doc)):
            if doc[i] not in G.nodes():
                G.add_node(doc[i])
            for j in range(i+1, i+window_size):
                if j < len(doc):
                    if G.has_edge(doc[i], doc[j]): G[doc[i]][doc[j]]['weight'] = G[doc[i]][doc[j]]['weight'] + 1
                    else: G.add_edge(doc[i], doc[j], weight = 1)

        #print(str(doc_count) + ' of ' + str(max_docs))

    return G

def create_graphs_of_words_dict(docs, window_size, words_dict):
    G = nx.Graph()
    max_docs = len(docs)
    doc_count = 0
    for doc in docs:
        doc_count = doc_count + 1
        if len(doc) == 0: continue
        for i in range(len(doc)):
            if words_dict[doc[i]] not in G.nodes():
                G.add_node(words_dict[doc[i]])
            for j in range(i+1, i+window_size):
                if j < len(doc):
                    if G.has_edge(words_dict[doc[i]], words_dict[doc[j]]): 
                        G[words_dict[doc[i]]][words_dict[doc[j]]]['weight'] = G[words_dict[doc[i]]][words_dict[doc[j]]]['weight'] + 1
                    else: 
                        G.add_edge(words_dict[doc[i]], words_dict[doc[j]], weight = 1)

        print(str(doc_count) + ' of ' + str(max_docs))

    return G

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	docs = load_file(args.input)
	docs = preprocessing(docs)

	words_dict = build_words_dict(docs)
	print("Vocabulary size: ", len(words_dict))

	with open('gr_vocab.txt', 'w') as f:
		for k, v in sorted(words_dict.items(), key=itemgetter(0)):
			f.write(str(k) + " " + str(v) + "\n")
		f.close()

	nx_G = create_graphs_of_words(docs, args.window_size)
	nx_G = nx.convert_node_labels_to_integers(nx_G, label_attribute='old_label')
	#nx_G = create_graphs_of_words_dict(docs, args.window_size, words_dict)

	G = node2vec.Graph(nx_G, False, args.p, args.q)

	print('Preprocessing transition probabilities...')
	G.preprocess_transition_probs()

	print('Simulating walks...')
	walks = G.simulate_walks(args.num_walks, args.walk_length)

	for i, walk in enumerate(walks):
		for j, word_id in enumerate(walk):
			walks[i][j] = nx_G.nodes[walk[j]]['old_label']

	print('Learning embeddings...')
	learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)

