#nvidia-smi
w2v_file='sougou_fullseg_300.vector'
#w2v_file='sougou_300.vector'
#w2v_file='xinhua_sent_300.vector'
#w2v_file='wiki.zh.text.vector'
#w2v_file='xinhua_sent_filter_300.vector'
cv=5
pkfile='mr.p'$cv
python process_data_rumor.py $w2v_file $pkfile $cv

python conv_net_sentence.py -nonstatic -word2vec $cv
