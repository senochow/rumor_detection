# -*- coding:utf8 -*-
import numpy as np
import cPickle
import json
import jieba
from collections import defaultdict
import sys, re
import pandas as pd
from sklearn.cross_validation import KFold
import ConfigParser

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

def load_extra_fea(filename):
    id_feature = {}
    with open(filename) as f:
        for line in f:
            #mid, label, fea = line.rstrip().split("\t")
            eid, mid, label, fea = line.rstrip().split("\t")
            id_feature[mid] = fea
    return id_feature

def load_json_file(filename, extra_fea_file, vocab, label, fold_map):
    revs = []
    id_extra_fea = load_extra_fea(extra_fea_file)
    fea_cnt = 0
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
                mid = weibo['mid'].encode('utf8', 'ignore')
                extra_fea = ''
                if mid in id_extra_fea:
                    extra_fea = id_extra_fea[mid]
                else:
                    fea_cnt += 1
                    continue
                    extra_fea = '0,0,0,0,0,0'
                cnt = 0
                #for word in ori_rev.split():
                for word in ori_rev.split():
                    cnt += 1
                    vocab[word] += 1
                event_index = num
                if label == 0:
                    event_index += 1000
                datum = {"y":label, "mid":mid ,"text": ori_rev, "extra_fea":extra_fea, "num_words": cnt, "split": fold_map[num], "event_id":event_index}
                revs.append(datum)
            num += 1
    print 'mid feature not found ', fea_cnt
    return revs

def build_data_cv(data_folder, extra_fea_folder, rumor_cnt, normal_cnt,cv=10, clean_string=True):
    """
    Loads data and split into cv folds.
    """
    fold_map_rumor = gen_kfold(rumor_cnt, cv)
    fold_map_normal = gen_kfold(normal_cnt, cv)

    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    rumor_datas = load_json_file(pos_file, extra_fea_folder[0], vocab, 1, fold_map_rumor)
    normal_datas = load_json_file(neg_file, extra_fea_folder[1], vocab, 0, fold_map_normal)
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
    nfold = int(sys.argv[2])
    conf_file = sys.argv[3]
    data_set = sys.argv[4] #which data set to use, 73 or 500

    cf = ConfigParser.ConfigParser()
    cf.read(conf_file)
    data_folder = [cf.get(data_set, "rumor_file"),cf.get(data_set, "normal_file")]
    extra_fea_folder = [cf.get(data_set, "extra_rumor_fea_file"), cf.get(data_set, "extra_normal_fea_file")]

    pkfile = cf.get(data_set, "pkfile"+str(nfold))
    rumor_cnt = int(cf.get(data_set, "rumor_cnt"))
    normal_cnt = int(cf.get(data_set, "normal_cnt"))
    print "loading data...",
    revs, vocab = build_data_cv(data_folder, extra_fea_folder, rumor_cnt, normal_cnt, cv=nfold, clean_string=True)
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
    # get idx to word map
    idx_word_map = {}
    for word, index in word_idx_map.items():
        if index in idx_word_map:
            print 'duplicates...'
        idx_word_map[index] = word
    idx_word_map[0] = 'NULL'
    cPickle.dump([revs, W, W2, word_idx_map, vocab, idx_word_map, max_l], open(pkfile, "wb"))
    print "dataset created!"
    #'''

