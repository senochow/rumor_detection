
import numpy as np

event_label = {}
event_fea = {}
f_event = open('event_index_fea.txt', 'w')
with open('event_feature') as f:
    for line in f:
        label, _, index, fea = line.rstrip().split(' ', 3)
        index = int(index)
        if index >= 100:
            index = 20
        f_event.write(str(index) + ' ' + fea+'\n')
        fea_list = [float(val) for val in fea.split()]
        if len(fea_list) != 300:
            print len(fea_list)
        event_fea.setdefault(index, [])
        event_fea[index].append(fea_list)
        event_label[index] = label
f_event.close()
# output res
fres = open('event_avg_feature.txt', 'w')
for index, feas in event_fea.items():
    avg_fea = np.mean(feas, axis=0)
    fres.write(' '.join([str(val) for val in avg_fea])+'\n')
    print event_label[index]
fres.close()
