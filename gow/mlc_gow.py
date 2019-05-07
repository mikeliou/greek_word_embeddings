import networkx as nx
import string
#import nltk
#from nltk.corpus import stopwords
#from datetime import datetime
#from collections import defaultdict
#from nltk.util import ngrams
#from nltk import bigrams
#from sys import maxsize
import matplotlib.pyplot as plt
import sys
import numpy as np
import re
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from argparse import ArgumentParser

def load_file(filename):
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            docs.append(line)

    return docs

def clean_str(s):
    s = re.sub(r"[^A-Za-zΑ-Ωα-ωΆ-Ώά-ώ0-9(),!?\'\`]", " ", s)     
    #string = re.sub(r"\'s", " \'s", string) 
    #string = re.sub(r"\'ve", " \'ve", string) 
    #string = re.sub(r"n\'t", " n\'t", string) 
    #string = re.sub(r"\'re", " \'re", string) 
    #string = re.sub(r"\'d", " \'d", string) 
    #string = re.sub(r"\'ll", " \'ll", string) 
    #string = re.sub(r",", " , ", string) 
    #string = re.sub(r"!", " ! ", string) 
    #string = re.sub(r"\(", " \( ", string) 
    #string = re.sub(r"\)", " \) ", string) 
    #string = re.sub(r"\?", " \? ", string) 
    #string = re.sub(r"\s{2,}", " ", string)

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

def create_graphs_of_words(docs, window_size):
    """ 
    Create graphs of words
    """
    #graphs = list()
    #sizes = list()
    #degs = list()
    #stop_words = set(stopwords.words('english'))

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
        #graphs.append(G)
        #sizes.append(G.number_of_nodes())
        #degs.append(2.0*G.number_of_edges()/G.number_of_nodes())

    #for graph in graphs:
    #pos = nx.spring_layout(G)
    #nx.draw(G, with_labels = True)
    #weights = nx.get_edge_attributes(G,'weight')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    #plt.show()

    #for a, b, data in sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True):
    #    print('{a} {b} {w}'.format(a=a, b=b, w=data['weight']))

    return G

def main():
    parser = ArgumentParser()
    parser.add_argument("-input", dest="input_file", default='datasets/testData.txt')
    parser.add_argument("-output", dest="output_file", default='test')
    parser.add_argument("-dim", dest="dimensions", default=100, type=int)
    parser.add_argument("-ws", dest="window_size", default=5, type=int)
    parser.add_argument("-minCount", dest="min_count", default=5, type=int)
    parser.add_argument("-sg", dest="sg", default=0, type=int)
    parser.add_argument("-numWalks", dest="num_walks", default=10, type=int)
    parser.add_argument("-walkLen", dest="walk_length", default=80, type=int)

    args = parser.parse_args()

    docs = load_file(args.input_file)
    docs = preprocessing(docs)

    words_dict = build_words_dict(docs)
    print("Vocabulary size: ", len(words_dict))

    #graph = create_graphs_of_words_dict(docs, args.window_size, words_dict)
    graph = create_graphs_of_words(docs, args.window_size)

    fh=open("gow.edgelist",'wb')
    nx.write_edgelist(graph, fh)

    node2vec = Node2Vec(graph, dimensions=args.dimensions, num_walks=args.num_walks, walk_length=args.walk_length)
    model = node2vec.fit(window=args.window_size, min_count=args.min_count, sg=args.sg)

    print(model.vocab)

    model.wv.save_word2vec_format(args.output_file + ".vec")
    model.save(args.output_file + ".bin")

    #print(model.wv.most_similar('battle'))
    
if __name__ == "__main__":
    main()