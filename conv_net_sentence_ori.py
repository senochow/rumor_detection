#! -*- coding:utf8 -*-
"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import time
warnings.filterwarnings("ignore")   

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
       
def cal_f1(pred, label):
    pos_ac, neg_ac = 0, 0
    pos_pred_num, neg_pred_num = 0, 0
    pos_num, neg_num = 0, 0
    for i in range(len(pred)):
        if pred[i] == 1:
            pos_pred_num += 1
        else:
            neg_pred_num += 1
        if label[i] == 1:
            pos_num += 1
            if label[i] == pred[i]:
                pos_ac += 1
        else:
            neg_num += 1
            if label[i] == pred[i]:
                neg_ac += 1
    pos_prec = float(pos_ac)/pos_pred_num
    neg_prec = float(neg_ac)/neg_pred_num

    pos_recall = float(pos_ac)/pos_num
    neg_recall = float(neg_ac)/neg_num

    print 'instance count : pos:' + str(pos_num) + '\t neg:' + str(neg_num)
    print 'hit count : pos:'+str(pos_ac) + '\t neg:' + str(neg_ac)
    print 'pos: precision: ' + str(pos_prec) + '\t recall:' + str(pos_recall)
    print 'neg: precision: ' + str(neg_prec) + '\t recall:' + str(neg_recall)
    sys.stdout.flush()
    #return float(pos_ac)/pos_num, float(neg_ac)/neg_num

def cal_event_prob(test_set_y, tmp_pred_prob, test_event_id):
    print 'cal event probability...'
    event_pred_list = {}
    event_label = {}
    for i in range(len(test_set_y)):
        index = test_event_id[i]
        event_pred_list.setdefault(index,[])
        event_pred_list[index].append(tmp_pred_prob[i][1])
        event_label[index] = test_set_y[i]
    event_pred = {}
    for index, preds in event_pred_list.items():
        pred_val = np.mean(preds)
        event_pred[index] = pred_val
    event_pred_label = {}
    for index, pred_val in event_pred.items():
        if pred_val > 0.5:
            event_pred_label[index] = 1
        else:
            event_pred_label[index] = 0
    print 'each event probability..'
    print event_pred
    avg_prec = cal_measure_info(event_label, event_pred_label)
    return avg_prec
        
def cal_event_mersure(test_set_y, tmp_pred_y, test_event_id):
    m_pred = {}
    event_label = {}
    for i in range(len(test_set_y)):
        index = test_event_id[i]
        m_pred.setdefault(index, [0,0])
        event_label[index] = test_set_y[i]
        m_pred[index][tmp_pred_y[i]] += 1
    # cal 
    print 'each event pred...'
    print m_pred
    event_pred = {}
    for index, preds in m_pred.items():
        if preds[0] > preds[1]:
            event_pred[index] = 0
        else:
            event_pred[index] = 1
    avg_prec = cal_measure_info(event_label, event_pred)
    return avg_prec

def cal_measure_info(event_label, event_pred):
    p_hit ,n_hit = 0, 0
    p_pred_num, n_pred_num = 0, 0
    p_num , n_num = 0, 0
    avg_prec = 0
    for index, label in event_label.items():
        pred_label = event_pred[index]
        if label == pred_label:
            avg_prec += 1
        if label == 1:
            p_num += 1
            if label == pred_label:
                p_hit += 1
                p_pred_num += 1
            else:
                n_pred_num += 1
        else:
            n_num += 1
            if label == pred_label:
                n_hit += 1
                n_pred_num += 1
            else:
                p_pred_num += 1
    avg_prec = 1.0*avg_prec/len(event_label)
    print 'hit count pos:%d, neg:%d'%(p_hit, n_hit)
    print 'precision : pos: %f, neg: %f' % (1.0*p_hit/p_pred_num, 1.0*n_hit/n_pred_num)
    print 'recall : pos: %f, neg: %f' % (1.0*p_hit/p_num, 1.0*n_hit/n_num)
    print 'average accuracy : %f'% (avg_prec)
    sys.stdout.flush() 
    return avg_prec



def train_conv_net(datasets,
                   test_event_id,
                   test_mid,
                   cv,
                   U,
                   img_w=300, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,2], 
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=5, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   extra_fea_len=6,
                   non_static=True):
    """
    datasets: 0 for tarin, 1 for test
    U: wordvec : {word_index: vector feature}
    Train a simple conv net
    img_h = sentence length (padded where necessary), 固定的长度：equal to max sentence length in dataset(pre computed)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes , 每一个filter 对于100个 feature map   
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-extra_fea_len-1  # last one is y
    print "img height ", img_h
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        # filter: conv shape, hidden layer: 就是最后的全连接层
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters    
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    # ??? set zero
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    # 转成 batch(50)*1*sent_len(134)*k(300)
    layer0_input = Words[T.cast(x[:,:img_h].flatten(),dtype="int32")].reshape((x.shape[0],1,img_h,Words.shape[1]))
    layer1_input_extra_fea = x[:,img_h:]
    conv_layers = []
    layer1_inputs = []
    # each filter has its own conv layer: the full conv layers = concatenate all layer to 1 
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    #layer1_inputs.append(layer1_input_extra_fea)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs) #+ extra_fea_len 
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random), 每次只取0.9倍的数据进行train， shuffle，另外的validation
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    #divide train set into train/val sets 
    test_set_x = datasets[1][:,:-1] 
    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]     
    train_set_x, train_set_y = shared_dataset((train_set[:,:-1],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:-1],val_set[:,-1]))
    n_val_batches = n_batches - n_train_batches
    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
            y: val_set_y[index * batch_size: (index + 1) * batch_size]}, allow_input_downcast=True)
            
    #compile theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]}, allow_input_downcast=True)               
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]}, allow_input_downcast=True) 
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x[:, :img_h].flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    test_layer1_input_extra_fea = x[:,img_h:]
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    #test_pred_layers.append(test_layer1_input_extra_fea)
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_y_pred_p = classifier.predict_p(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,y], test_error, allow_input_downcast=True)   
    test_model_f1 = theano.function([x], test_y_pred, allow_input_downcast=True)  
    test_model_prob = theano.function([x], test_y_pred_p, allow_input_downcast=True)  
    test_layer1_feature = theano.function([x], test_layer1_input, allow_input_downcast=True)
    test_extra_fea = theano.function([x], test_layer1_input_extra_fea, allow_input_downcast=True)
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    fp = 0
    avg_precsion = 0
    while (epoch < n_epochs):        
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)                   
        print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.))
        test_loss = test_model_all(test_set_x,test_set_y)
        print 'cur precision: ', 1- test_loss
        #tmp_pred_prob = test_model_prob(test_set_x)
        #cal_event_prob(test_set_y, tmp_pred_prob, test_event_id)
        #tmp_feature = test_layer1_feature(test_set_x)
        #cal_f1(tmp_pred_y, test_set_y)
        if epoch == n_epochs:
            #tmp_pred_y = test_model_f1(test_set_x)
            tmp_pred_prob = test_model_prob(test_set_x)
            test_loss = test_model_all(test_set_x,test_set_y)
            fp = 1- test_loss
            print 'last : ', fp
            #cal_f1(tmp_pred_y, test_set_y)
            #if tmp_perf > test_perf:
            #avg_precsion1 = cal_event_mersure(test_set_y, tmp_pred_y, test_event_id)
            avg_precsion = cal_event_prob(test_set_y, tmp_pred_prob, test_event_id)
            #for i in range(len(tmp_pred_y)):
            #    print '%d %d %d %s' % (test_set_y[i], tmp_pred_y[i], test_event_id[i], ' '.join([str(val) for val in tmp_feature[i]]))
            '''
            tmp_feature = test_layer1_feature(test_set_x)
            with open("sentence_feature_cv"+str(cv)+".txt", "w") as f:
                for i in range(len(tmp_feature)):
                    f.write(str(test_event_id[i])+"\t"+test_mid[i]+"\t"+','.join([str(val) for val in tmp_feature[i]])+"\n")
            '''
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss = test_model_all(test_set_x,test_set_y)        
            test_perf = 1- test_loss       
        
    return test_perf, fp, avg_precsion

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes, (lastword+pad=filter_h).
        sent = pad + sentence + pad , pad = filter_h-1
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    test_mid = []
    test_event_id = []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        # add extra feature to sent fea, word_index(max_l) + extra_fea + y
        extra_fea = [int(val) for val in rev["extra_fea"].split(",")]
        sent += extra_fea
        mid = rev["mid"]
        # if the sententce if larger than max_l, then use the (0, max_l)
        #if len(sent) > max_l:
        #    sent = sent[:max_l]
        sent.append(rev["y"])
        if rev["split"]==cv:            
            test.append(sent)
            test_event_id.append(rev["event_id"])
            test_mid.append(mid)
        else:  
            train.append(sent)   
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    print 'traing set :\t', len(train)
    print 'testing set :\t', len(test)
    return [train, test], test_event_id, test_mid
  
   
if __name__=="__main__":
    print "loading data...",
    mode= sys.argv[1]
    word_vectors = sys.argv[2]    
    cv = int(sys.argv[3])
    x = cPickle.load(open("mr.p"+str(cv),"rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"

    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("conv_net_classes.py")    
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W
    results = []
    fres = []
    avg_plist = []
    r = range(0,cv)    
    start_time = time.time()
    for i in r:
        datasets, test_event_id, test_mid = make_idx_data_cv(revs, word_idx_map, i, max_l=133,k=300, filter_h=9)
        #datasets, test_event_id = make_idx_data_cv(revs, word_idx_map, i, max_l=133,k=300, filter_h=9)
        perf, fp, avg_precsion = train_conv_net(datasets,
                              test_event_id,
                              test_mid,
                              i,
                              U,
                              lr_decay=0.95,
                              filter_hs=[7,8,9],
                              #filter_hs=[3,4,5],
                              conv_non_linear="relu",
                              hidden_units=[100,2], 
                              shuffle_batch=True, 
                              n_epochs=5, 
                              sqr_norm_lim=9,
                              non_static=non_static,
                              batch_size=50,
                              dropout_rate=[0.5])
        print "cv: " + str(i) + ", perf: " + str(perf)
        results.append(perf)  
        fres.append(fp)
        avg_plist.append(avg_precsion)
        #break
    print 'total time : %.2f minutes' % ((time.time()-start_time)/60)
    print str(np.mean(results))
    print str(np.mean(fres))
    print 'all avg precision : prec: %f' % (np.mean(avg_plist))
