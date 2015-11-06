import numpy as np
import time
import theano
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
    def load(cls, label_map_file, chr_map_file, label_file, post_file):
        t_start = time.time()
        cls.label_keys = np.genfromtxt(label_map_file, usecols=0, dtype=str)
        for i in xrange(len(cls.label_keys)): cls.label_num[cls.label_keys[i]] = i
        label_raw_data = np.genfromtxt(label_map_file, usecols=[1],dtype=str)
        #cls.label_map = {key: row for key, row in zip(cls.label_keys, label_raw_data)}  # not compatible with python 2.6.6
        for key, row in zip(cls.label_keys, label_raw_data):
            cls.label_map[key] = row

        cls.chr_keys = np.genfromtxt(chr_map_file, usecols=0, dtype=str)
        chr_raw_data = np.genfromtxt(chr_map_file, usecols=[1,2],dtype=None)
        #cls.chr_map = {key: row for key, row in zip(cls.chr_keys, chr_raw_data)}  # not compatible with python 2.6.6
        for key, row in zip(cls.chr_keys, chr_raw_data):
            cls.chr_map[key] = row
        
        with open(label_file, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                token = line.split(',')
                sample_id = token[0]
                cls.label[sample_id] = token[1]
                cls.label_ans.append(token[1])

        with open(post_file, 'r') as f:
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
        #return cls.x_seq , cls.y_hat

    @classmethod
    def load_y(cls, label):
        y = [0]*48
        #y[ cls.chr_map[cls.label_map[label]][0] ] = 1
        y[ cls.label_num[label] ] = 1
        return y

    @classmethod
    def load_test_data(cls, fname):
        print 'loading test data...'
        t_start = time.time()
        test_matrix={}
        test_id=[]
        ans={}
        tmp_ans=[]
        tmp=[]
        with open(fname, 'r') as f:
            frame = ''
            id_index = -1
            for line in f:
                #print(line)
                line = line.replace('\n', '')
                token = line.split(' ')
                index = token[0].rfind('_')

                if not frame == token[0][0:index]:
                    frame = token[0][0:index]
                    test_id.append(frame)
                    id_index = id_index + 1
                    tmp.append([])
                    tmp_ans.append([])

                attr = []
                for attr_str in token[1:]: attr.append(float(attr_str))
                tmp[id_index].append(attr)
                test_matrix[frame] = tmp[id_index]

                tmp_ans[id_index].append(token[0])
                ans[frame] = tmp_ans[id_index]
        
        '''
        cls.label_keys = np.genfromtxt('../dataset/phones/48_39.map', usecols=0, dtype=str)
        with open('../dataset/label/train.lab', 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                token = line.split(',')
                sample_id = token[0]
                cls.label[sample_id] = token[1]
                cls.label_ans.append(token[1])
        '''

        t_end = time.time()
        print 'test data loaded. loading time: %f sec' % (t_end - t_start)
        return test_matrix, test_id, ans

    @classmethod
    def load_matrix(cls, m=None, n=None, name=None, fname=None):
        print('init parameters')
        if fname:
            with open(fname, 'r') as f:
                if name:
                    w_all = []
                    line = f.readline() 
                    token = line.split(' ')
                    for i in xrange(len(token)):
                        if token[i] == name:
                            data = f.read()
                            data_token = data.split('\n')
                            matrix_data = data_token[2*i-1].split(',')
                            num_data = data_token[2*i-2].split(',')
                            data_x = int(num_data[0])
                            data_y = int('0'+num_data[1][1:])
                            if data_y != 0:
                                #print(data_y)
                                for x in range(data_x):
                                    w = []
                                    for y in range(data_y):
                                       w.append(float(matrix_data[x*data_y+y]))
                                    w_all.append(w)
                            else:
                                w = []
                                for x in range(data_x):
                                    w.append(float(matrix_data[x]))
                                w_all = w
                            return theano.shared(np.array(w_all).astype(theano.config.floatX),name)
                else:
                    w_all_matrix = []
                    line = f.readline()
                    token = line.split(' ')
                    data = f.read()
                    data_token = data.split('\n')           
                    i=0                 
                    while i < len(token)-2:
                        w_all = []
                        #print(i)
                        matrix_data = data_token[2*i+1].split(',')
                        num_data = data_token[2*i].split(',')
                        #print(data_token[2*i])
                        data_x = int(num_data[0])
                        data_y = int('0'+num_data[1][1:])
                        if data_y != 0:
                            #print(data_y)
                            for x in range(data_x):
                                w = []
                                for y in range(data_y):
                                   w.append(float(matrix_data[x*data_y+y]))
                                w_all.append(w)
                        else:
                            w = []
                            for x in range(data_x):
                                w.append(float(matrix_data[x]))
                            w_all = w
                        w_all_matrix.append(theano.shared(np.array(w_all).astype(theano.config.floatX),token[i+1]))
                        i+=1
                return w_all_matrix                     
                # with open(fname, w):
                # load file into theano shared variablie
                # return theano.shared( np().astype(theano.config.floatX) )
            pass
        else:
            if n:
                return theano.shared( np.eye(m, n).astype(theano.config.floatX), name)
            else:
                return theano.shared( np.zeros(m).astype(theano.config.floatX), name)

    @classmethod
    def save_parameters(cls, parameters):
        # save parameters in a way that the data can be easily loaded by theano
        #np.set_printoptions(threshold=2**32)
        print('save all parameters')
        with open('rnn_parameter.txt', 'w') as f:
            f.write('parameters: ')
            for p in parameters:
                f.write( p.name +' ')
            f.write('\n')
            for p in parameters:
                t = str(p.get_value().shape).replace("(",'').replace(")",'')
                t = str(t).replace("L",'')
                f.write(t + '\n')
                t = p.get_value().tolist()
                t = str(t).replace("[",'')
                t = str(t).replace("]",'')
                f.write("%s\n" %t )
    
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

