# -*- coding:utf-8 -*-
############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:

Authors: zhouxing(@ict.ac.cn)
Date:    2016/03/04 09:54:23
File:    split_events.py
"""
import sys

events_file = sys.argv[1]
f = open(events_file)
dirname= sys.argv[2]

while 1:
    line = f.readline()
    if not line:
        break
    index, prob, content = line.rstrip().split("\t")
    tmp_f = open(dirname+"/event_"+index+"_keywords.txt", "a")
    tmp_f.write(line)
    for i in range(6):
        line = f.readline()
        tmp_f.write(line)
    tmp_f.write("\n")
    tmp_f.close()

f.close()

#for key,val in all_f.items():
#    all_f[key].close()

# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
