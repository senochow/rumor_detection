

CNN framework for rumor detection
===

workflow:
===

Text->(word embedding through word2vec)-> convolution -> max pooling -> sentence feature + extra feature layer -> softmax 

1. word embedding : use word2vec for training, weibo data filter length by 10, 5kw weibo
2. conv-net: use multiple filter size, each filter can get one feature through max-pooling 
   keyphrase extraction: for each filter size, get the most selected phrase
3. feature combination: 300 sentence level feature + sentiment and word entitiy faeture


code structure
===
preprocess: CNNPreprocess.java , extra_feature: WeiboFeature/WeiboFeatureExtrator.java

1. process_data_rumor.py
 - input: word2vec file (pre trained on large scale data set), pkfile, nfold
 - data_folder: weibo messages by split words
 - extra_fea: selected feature by IG, mid, feature
 * word not in word2vec, initialize by uniform(-0.25,0.25)
 * vocabulary: 0 for NULL, and others start from 1 (0 also used for padding null words)
 * output pkfile: sentenses, word2vec, random_vectors, word->id, vocab, id->word
 * fist process must get the max_length and set for cnn

Tricks
===
1. min-batch for training, and each epoch use only 90% data for trianing
2. shuffle batch
3. weight initialize
4. adadelta for update weight
5. dropout for hidden layer
