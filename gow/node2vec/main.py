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
import re
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

	parser.add_argument('--iter', default=1, type=int,
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
    
    for doc in docs:
        preprocessed_docs.append(clean_str(doc))

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
    max_docs = len(docs)
    doc_count = 0
    for doc in docs:
        doc_count = doc_count + 1
        if len(doc) == 0: continue
        for i in range(len(doc)):
            if doc[i] not in G.nodes():
                G.add_node(doc[i])
            for j in range(i+1, i+window_size):
                if j < len(doc):
                    if G.has_edge(doc[i], doc[j]): G[doc[i]][doc[j]]['weight'] = G[doc[i]][doc[j]]['weight'] + 1
                    else: G.add_edge(doc[i], doc[j], weight = 1)

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

	nx_G = create_graphs_of_words(docs, args.window_size)

	G = node2vec.Graph(nx_G, False, args.p, args.q)

	print('Preprocessing transition probabilities...')
	G.preprocess_transition_probs()

	print('Simulating walks...')
	walks = G.simulate_walks(args.num_walks, args.walk_length)

	learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)

