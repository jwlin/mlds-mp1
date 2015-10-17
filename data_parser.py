# vim: set ts=4 sw=4 et: -*- coding: utf-8 -*-

import itertools
import random
import time
import theano
import numpy as np


class DataParser:
    sample = {}  # e.g. {'mzmb0_sx356_40': [83.17422, -10.78858, -19.54186, 14.11323, ...], ...}
    label = {}  # e.g. {'mzmb0_sx356_40': 'ae', ... }
    label_index = []  # e.g. ['ae', 'cl', ...]
    label_map = {}  # e.g. {'ae': 'ae', 'cl': 'sil', ...}
    dimension_x = 0
    dimension_y = 0

    @classmethod
    def load(cls, sample_file, label_file, label_map_file):
        t_start = time.time()
        print 'loading data...'

        with open(sample_file, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                token = line.split(' ')
                sample_id = token[0]
                attr = []
                for attr_str in token[1:]:
                    attr.append(float(attr_str))
                cls.sample[sample_id] = attr
        cls.dimension_x = len(cls.sample.itervalues().next())

        with open(label_file, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                token = line.split(',')
                sample_id = token[0]
                cls.label[sample_id] = token[1]
            
        with open(label_map_file, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                token = line.split('\t')
                label_48 = token[0]
                cls.label_map[label_48] = token[1]
                cls.label_index.append(label_48)
        cls.dimension_y = len(cls.label_index)
        
        t_end = time.time()
        print 'data loaded. loading time: %f sec' % (t_end - t_start)

    @classmethod
    def test(cls):
        print 'data integrity check...'
        sample_size = 10000
        assert len(cls.sample)==sample_size
        assert len(cls.sample)==len(cls.label)
        assert cls.dimension_x==39
        #assert cls.dimension_x==69
        assert cls.dimension_y==48
        #print cls.label_index
        #print 'oracle:'
        #print 'mzmb0_sx356_315 34.80573 -30.16864 -2.489416 -7.240193 -6.575228 2.194885 -5.885315 -5.310679 -0.9152679 -2.394212 3.871122 -6.066998 -3.769938 0.03159332 0.4700251 1.014741 -0.4032825 0.5082797 0.8969792 2.137414 2.388781 1.270098 -1.145748 -5.196232 -0.4523884 -3.496056 -0.009477973 -0.2245679 -0.6435395 -0.2106031 -0.03572752 -0.3485634 0.3479201 0.08721232 0.467126 0.2610434 -0.359623 -0.31955 0.446179'
        #print 'data:'
        #print cls.sample['mzmb0_sx356_315']
        assert cls.label['mcdr0_sx254_162'] == 'er'
        assert cls.label_map[cls.label['mcdr0_sx254_162']] == 'er'
        assert cls.label['mrfl0_sx256_150'] == 'cl'
        assert cls.label_map[cls.label['mrfl0_sx256_150']] == 'sil'
        print 'data integrity check done.'
        print 'make_batch() test ...'
        batch_size = 50
        x_batch, y_batch, batch_num = DataParser.make_batch(batch_size)
        assert batch_num == int(sample_size/batch_size)
        assert len(x_batch) == batch_num
        assert len(y_batch) == batch_num
        x_t = map(list, zip(*x_batch[-2]))  # transpose: dimension_x*batch_size --> batch_size*dimension_x
        #print x_t[-1]
        assert len(x_t[-1]) == cls.dimension_x
        y_t = map(list, zip(*y_batch[-2]))  # transpose: dimension_y*batch_size --> batch_size*dimension_y
        assert len(y_t[-1]) == cls.dimension_y
        assert y_t[-1].count(1) == 1
        #print cls.label_index[y_t[-1].index(1)]
        print 'make_batch() test done'
        

    @classmethod
    def make_batch(cls, batch_size):  # randomly divide samples and labels into groups of batch_size
        assert not (len(cls.sample) % batch_size), 'can not divide samples with ' + str(batch_size)
        t_start = time.time()
        print 'making batch...'
        sample_ids = list(cls.sample.keys())
        random.shuffle(sample_ids)
        whole_x_batch = []
        whole_y_batch = []
        batch_num = int(len(cls.sample)/batch_size)
        for i in xrange(batch_num):
            mini_x_batch = []
            mini_y_batch = []
            for j in xrange(cls.dimension_x):
                mini_x_batch.append([])
            for j in xrange(cls.dimension_y):
                mini_y_batch.append([])
            for k in xrange(batch_size):
                sample_id = sample_ids.pop()
                for l in xrange(cls.dimension_x):
                    mini_x_batch[l].append(cls.sample[sample_id][l])
                for l in xrange(cls.dimension_y):
                    label = cls.label[sample_id]
                    mini_y_batch[l].append(1 if l==cls.label_index.index(label) else 0)
            whole_x_batch.append(mini_x_batch)
            whole_y_batch.append(mini_y_batch)
        assert len(sample_ids)==0
            
        '''
        print len(whole_x_batch), len(whole_y_batch)
        for e in whole_x_batch[-2]:
            print e
        for e in whole_y_batch[-2]:
            print e
        '''
        t_end = time.time()
        print 'batch made. elapsed time: %f sec' % (t_end - t_start)
        return whole_x_batch, whole_y_batch, batch_num

    @classmethod
    def load_test_data(cls, fname):
        t_start = time.time()
        print 'loading test data...'
        test_id = []
        test_matrix = []
        with open(fname, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                token = line.split(' ')
                sample_id = token[0]
                test_id.append(sample_id)
                attr = []
                for attr_str in token[1:]:
                    attr.append(float(attr_str))
                test_matrix.append(attr)
        test_matrix = map(list, zip(*test_matrix))  # transpose
        t_end = time.time()
        print 'test data loaded. loading time: %f sec' % (t_end - t_start)
        return test_matrix, test_id

    @classmethod
    def load_validation_data(cls, fname, lab_fname):
        t_start = time.time()
        print 'loading validation data...'
        validation_matrix = []
        validation_y = []
        label = {}
        with open(lab_fname, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                token = line.split(',')
                sample_id = token[0]
                label[sample_id] = token[1]
        with open(fname, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                token = line.split(' ')
                sample_id = token[0]
                attr = []
                for attr_str in token[1:]:
                    attr.append(float(attr_str))
                validation_matrix.append(attr)
                label_array = []
                tag = label[sample_id]
                for i in xrange(cls.dimension_y):
                    label_array.append(1 if i==cls.label_index.index(tag) else 0)
                validation_y.append(label_array)
        validation_matrix = map(list, zip(*validation_matrix))  # transpose
        validation_y = map(list, zip(*validation_y))  # transpose
        t_end = time.time()
        print 'validation data loaded. loading time: %f sec' % (t_end - t_start)
        return validation_matrix, validation_y
    
    @classmethod
    def check(cls, fname1, fname2):
        map1 = {}
        map2 = {}
        with open(fname1, 'r') as f1:
            for line in f1:
                line = line.replace('\n', '')
                token = line.split(',')
                map1[token[0]] = token[1]
        with open(fname2, 'r') as f2:
            for line in f2:
                line = line.replace('\n', '')
                token = line.split(',')
                map2[token[0]] = token[1]
        assert len(map1)==len(map2)
        count = 0
        for key in map1.keys():
            if map1[key] == map2[key]:
                count += 1
        return count, len(map1)

    @classmethod
    def load_matrix(cls, m, n=None, fname=None):
        if fname:
            # with open(fname, w):
                # load file into theano shared variablie
                # return theano.shared(()).astype(np.float64) )
            pass
        else:
            if n:
                return theano.shared( np.random.randn(m, n).astype(np.float64) )
            else:
                return theano.shared( np.random.randn(m).astype(np.float64) )

    @classmethod
    def save_parameters(cls, parameters):
        '''
        # save parameters in a way that the data can be easily loaded by theano
        np.set_printoptions(threshold=2**32)
        with open('parameter.txt', 'w') as f:
            f.write('parameters: w1, w2, w3, b1, b2, b3\n')
            for p in parameters:
                f.write(str(p.get_value().shape) + '\n')
                f.write(str(p.get_value()) + '\n')
        '''
        pass

if __name__ == '__main__':
    sample_file = 'mfcc/train-1.ark'  # separated by blank
    label_file = 'label/train-1.lab'  # separated by comma
    #sample_file = 'mfcc/train_sample.ark'  # separated by blank
    #label_file = 'label/train_sample.lab'  # separated by comma
    label_map_file = 'phones/48_39.map'  # separated by tab
    DataParser.load(sample_file, label_file, label_map_file)
    DataParser.test()
    
    #sample_size = 495
    #sample_test_id = 'mzmb0_sx86_215'
    matrix, ids = DataParser.load_test_data(sample_file)
    assert len(ids) == len(DataParser.sample)
    #assert ids[-3] == sample_test_id
    assert len(matrix) == DataParser.dimension_x
    m2 = matrix[-2]
    #assert len(m2) == sample_size
    #print m2
    correct, all_record = DataParser.check(label_file, label_file)
    assert correct==all_record

    #DataParser.load_validation_data(sample_file, label_file)

