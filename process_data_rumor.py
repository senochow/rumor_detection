# -*- coding:utf8 -*-
import numpy as np
import cPickle
import json
import jieba
from collections import defaultdict
import sys, re
import pandas as pd
from sklearn.cross_validation import KFold

def gen_kfold(nums, cv):
    kf = KFold(nums, n_folds=cv)
    index = 0
    fold_map = {}
    for train, test in kf:
        for i in test:
            fold_map[i] = index
        index += 1
    print index
    return fold_map

def load_json_file(filename, vocab, label, fold_map):
    revs = []
    with open(filename) as f:
        num = 0
        for line in f:
            d = json.loads(line.rstrip())   
            weibos = d['weibos']
            e_name = d['event_name'].encode('utf8', 'ignore')
            #if label == 1:
            #    print str(num) + '\t' + e_name
            #else:
            #    print str(num+100) + '\t' + e_name
            for weibo in weibos:
                content = weibo['content']#.encode("utf8")
                #words = jieba.cut(content)
                #words = weibo['words'].split(" ")
                #print words
                # words str, connect by space
                ori_rev = weibo['words']
                cnt = 0
                #for word in ori_rev.split():
                for word in ori_rev.split():
                    cnt += 1
                    vocab[word] += 1
                event_index = num
                if label == 0:
                    event_index += 100
                datum = {"y":label, "text": ori_rev, "num_words": cnt, "split": fold_map[num], "event_id":event_index}
                revs.append(datum)
            num += 1
    return revs

def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 5 folds.
    """
    fold_map = gen_kfold(73, cv)
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    rumor_datas = load_json_file(pos_file, vocab, 1, fold_map)
    normal_datas = load_json_file(neg_file, vocab, 0, fold_map)
    revs = rumor_datas + normal_datas
    return revs, vocab
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_txt_vec(filename, vocab):
    print len(vocab)
    
    word_vecs = {}
    f = open(filename)
    line = f.readline().rstrip()
    vocab_size, layer1_size = line.split()
    while 1:
        line = f.readline()
        if not line:
            break
        line = line.rstrip().decode("utf8", 'ignore')
        word, fea = line.split(' ', 1)
        if word in vocab:
            #print word
            word_vecs[word] = np.fromstring(fea, dtype='float32', sep=' ')
    return word_vecs

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()


if __name__=="__main__":    
    w2v_file = sys.argv[1]     
    pkfile = sys.argv[2]
    nfold = int(sys.argv[3])
    #data_folder = ["data/rumor_events_messages.json","data/normal_events_messages.json"]
    data_folder = ["data/rumor_events_messages_words.json","data/normal_events_messages_words.json"]
    print "loading data...",        
    revs, vocab = build_data_cv(data_folder, cv=nfold, clean_string=True)
    #'''
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    mean_l = np.mean(pd.DataFrame(revs)["num_words"])
    min_l = np.min(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    print 'mean, max' + str(mean_l) + ' \t' + str(min_l)
    
    w2v = load_txt_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open(pkfile, "wb"))
    print "dataset created!"
    #'''

