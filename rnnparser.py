import numpy as np
import time

class TestParser:
    x_seq=[]
    y_hat=[]
    label = {}
    label_ans = []

    label_map = {}
    label_keys = []
    chr_map = {}
    chr_keys = []
    dimension_x = 0
    dimension_y = 0
 
    @classmethod
    def load(cls):
        t_start = time.time()
        cls.label_keys = np.genfromtxt('phones/48_39.map', usecols=0, dtype=str)
        label_raw_data = np.genfromtxt('phones/48_39.map', usecols=[1],dtype=str)
        cls.label_map = {key: row for key, row in zip(cls.label_keys, label_raw_data)}

        cls.chr_keys = np.genfromtxt('48_idx_chr.map_b', usecols=0, dtype=str)
        chr_raw_data = np.genfromtxt('48_idx_chr.map_b', usecols=[1,2],dtype=None)
        cls.chr_map = {key: row for key, row in zip(cls.chr_keys, chr_raw_data)}
        
        with open('label/train.lab', 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                token = line.split(',')
                sample_id = token[0]
                cls.label[sample_id] = token[1]
                cls.label_ans.append(token[1])

        with open('posteriorgram/train.post', 'r') as f:
            frame = ''
            id_index = -1
            for line in f:
                #print(line)
                line = line.replace('\n', '')
                token = line.split(' ')
                index = token[0].rfind('_')
                y = TestParser.load_y(cls.label[token[0]])

                if not frame == token[0][0:index]:
                    frame = token[0][0:index]
                    id_index = id_index + 1
                    cls.x_seq.append([])
                    cls.y_hat.append([])

                attr = []
                for attr_str in token[1:]: attr.append(float(attr_str))
                cls.x_seq[id_index].append(attr)
                cls.y_hat[id_index].append(y)
        cls.dimension_x = len(TestParser.x_seq[0][0])
        cls.dimension_y = len(TestParser.y_hat[0][0])

        t_end = time.time()
        print 'data loaded. loading time: %f sec' % (t_end - t_start)
        return cls.x_seq , cls.y_hat
    @classmethod
    def load_y(cls, label):
        y = [0]*48
        y[ cls.chr_map[cls.label_map[label]][0] ] = 1
        return y
'''
TestParser.load()
'''