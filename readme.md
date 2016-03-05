

CNN framework for rumor detection
===

workflow:
===

Text->(word embedding through word2vec)-> convolution -> max pooling -> sentence feature + extra feature layer -> softmax 

1. word embedding : use word2vec for training, weibo data filter length by 10, 5kw weibo
2. conv-net: use multiple filter size, each filter can get one feature through max-pooling 
   keyphrase extraction: for each filter size, get the most selected phrase
3. feature combination: 300 sentence level feature + sentiment and word entitiy faeture


Tricks
===
1. min-batch for training, and each epoch use only 90% data for trianing
2. shuffle batch
3. weight initialize
4. adadelta for update weight
