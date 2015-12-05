from hmm import Hmm
import time

label_map_file = '../dataset/phones/48_39.map'
chr_map_file = '../dataset/48_idx_chr.map_b'
label_file = '../dataset/label/train.lab'
post_file = '../dataset/posteriorgram/test6.post'#'../dataset/posteriorgram/train2.post'
Hmm.loadgetprob(label_map_file, chr_map_file, label_file, post_file)#'test_y_.txt'


setting = 0.57
for i in xrange(3):
    n, p, path = Hmm.viterbi(setting)
    #count,counttotal = Hmm.check(path)
    #print (str(count) +'/'+ str(counttotal))
    Hmm.save_result(path, setting)
    setting += 0.05
