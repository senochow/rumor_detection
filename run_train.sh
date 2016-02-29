#nvidia-smi
dirname='/media/Data/zhouxing/data/word_vectors/'
#dirname='./word_vectors/'
#w2v_file=${dirname}'sougou_fullseg_300.vector'
w2v_file=${dirname}'weibo_filter_5kw_300.vector'
#w2v_file=${dirname}'c_w2v_1y_filter_weibo.vector'
#w2v_file=${dirname}'c_w2v_1y_filter_weibo_cbow.vector'
#w2v_file=${dirname}'c_w2v_1y_filter_weibo_cbow_iter3.vector'
#w2v_file=${dirname}'c_w2v_all_filter_weibo_skip_iter1.vector'
#w2v_file=${dirname}'c_w2v_all_filter_weibo_cbow_iter10.vector'
#w2v_file=${dirname}'c_w2v_all_filter_weibo_skip_iter5.vector'
#w2v_file=${dirname}'weibo_filter_5kw_part2_300.vector'
#w2v_file='/home/zhouxing/tools/word2vec/src/all_filter_weibo.vector'
#w2v_file=${dirname}'sougou_300.vector'
#w2v_file=${dirname}'xinhua_sent_300.vector'
#w2v_file=${dirname}'wiki.zh.text.vector'
#w2v_file=${dirname}'xinhua_sent_filter_300.vector'
cv=5
pkfile='mr.p'$cv
#python process_data_rumor.py $w2v_file $pkfile $cv
#python process_data_17.py $w2v_file $pkfile $cv

#python conv_net_sentence.py -nonstatic -word2vec $cv
python conv_net_sentence.py -static -word2vec $cv
#python conv_net_sentence_ori.py -static -word2vec $cv
