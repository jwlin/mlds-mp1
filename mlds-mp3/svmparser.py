import numpy as np
import time
#import theano
import random

class TestParser:
    x_seq=[]
    y_hat=[]
    label = {}
    label_ans = []

    label_map = {}
    label_keys = []
    label_num={}
    chr_map = {}
    chr_keys = []
    dimension_x = 0
    dimension_y = 0
        
    @classmethod
    def load(cls, label_map_file, chr_map_file, train_file, label_file):
        t_start = time.time()
        cls.label_keys = np.genfromtxt(label_map_file, usecols=0, dtype=str)
        for i in xrange(len(cls.label_keys)): 
            cls.label_num[cls.label_keys[i]] = i
        label_raw_data = np.genfromtxt(label_map_file, usecols=[1],dtype=str)
        for key, row in zip(cls.label_keys, label_raw_data):
            cls.label_map[key] = row

        cls.chr_keys = np.genfromtxt(chr_map_file, usecols=0, dtype=str)
        #chr_raw_data = np.genfromtxt(chr_map_file, usecols=[1,2],dtype=None)
        chr_raw_data = np.genfromtxt(chr_map_file, usecols=[2],dtype=str)
        for key, row in zip(cls.chr_keys, chr_raw_data):
            cls.chr_map[key] = row
        
        if label_file:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.replace('\n', '')
                    token = line.split(',')
                    sample_id = token[0]
                    cls.label[sample_id] = token[1]
                    cls.label_ans.append(token[1])

        with open(train_file, 'r') as f:
            frame = ''
            id_index = -1
            for line in f:
                line = line.replace('\n', '')
                token = line.split(' ')
                index = token[0].rfind('_')
                if label_file:
                    y = cls.label_num[ cls.label[token[0]] ]
                else:
                    y = 0
                if not frame == token[0][0:index]:
                    frame = token[0][0:index]
                    id_index = id_index + 1
                    cls.x_seq.append([])
                    cls.y_hat.append([])

                attr = []
                for attr_str in token[1:]: 
                    attr.append(float(attr_str))
                cls.x_seq[id_index].append(attr)
                cls.y_hat[id_index].append(y)
        cls.dimension_x = len(TestParser.x_seq[0][0])
        #cls.dimension_y = len(TestParser.y_hat[0])

        t_end = time.time()
        print 'data loaded. loading time: %f sec' % (t_end - t_start)

    @classmethod
    def trim(cls, seq):
        # trim leading and tailing 'L' (sil)
        while (seq.startswith('L')):
            seq = seq[1:]
        while (seq.endswith('L')):
            seq = seq[:-1]
        
        # zip duplicate letters
        zipped = ''
        for i in xrange(len(seq)):
            if i==0:
                zipped = seq[i]
            elif seq[i] == zipped[-1]:
                continue
            else:
                zipped += seq[i]
        return zipped


def transition_prob_from_data(label_file, label_map_file):
    dim_y = 48
    label = {}
    label_keys = []
    label_num={}    
    
    t_start = time.time()
    print 'calculating trasition prob from data...'
    label_keys = np.genfromtxt(label_map_file, usecols=0, dtype=str)
    for i in xrange(len(label_keys)): 
        label_num[label_keys[i]] = i
        
    tran_vec = [0]*(48*48)
    with open(label_file, 'r') as f:
        prev_utterance = ''
        prev_y = -1
        for line in f:
            line = line.replace('\n', '')
            token = line.split(',')
            utterance_spliter = token[0].rfind('_')
            y = label_num[token[1]]
            
            if not prev_utterance == token[0][0:utterance_spliter]:
                prev_y = y
                prev_utterance = token[0][0:utterance_spliter]
            else:
                tran_vec[48*prev_y+y] += 1
                prev_y = y

        tran_vec_prob = []
        for i in xrange(48):
            state_tran_vec = tran_vec[i*48:(i+1)*48]
            state_tran_sum = sum(state_tran_vec)
            state_tran_vec = [float(e)/state_tran_sum for e in state_tran_vec]
            tran_vec_prob += state_tran_vec

        print len(tran_vec_prob), tran_vec_prob
        #tran_vec = np.log(tran_vec_prob)
        #print len(tran_vec_prob), tran_vec_prob
    
    t_end = time.time()
    print 'transition table calculated. loading time: %f sec' % (t_end - t_start)

