# vim: set ts=4 sw=4 et: -*- coding: utf-8 -*-

import paths
from svmparser import TestParser
#import numpy
import Levenshtein
#import cPickle
#import csv

def read():
    TestParser.load(paths.label_map_file, paths.chr_map_file, paths.label_file, paths.mfcc_train)
    print TestParser.dimension_x
    print TestParser.x_seq[-2][-3], TestParser.y_hat[-2][-3]

    examples = []
    assert len(TestParser.x_seq) == len(TestParser.y_hat)
    for i in xrange(len(TestParser.x_seq)):
        assert len(TestParser.x_seq[i]) == len(TestParser.y_hat[i])
        tup = (TestParser.x_seq[i], TestParser.y_hat[i])
        examples.append(tup)
    assert TestParser.x_seq[-3][-3] == examples[-3][0][-3]

def psi(x, y):
    prev_num = -1
    feature_vector = [0]*(39*48+48*48)
    for i in xrange(len(x)):
        offset = y[i]*39
        for j in xrange(39):
            feature_vector[offset+j] += x[i][j]
        base = 39*48
        if prev_num >= 0:
            offset = prev_num*48
            feature_vector[base + offset + y[i]] = 1  # or +=1?
        prev_num = y[i]
    '''
    # do sparse format
    sparse_vec = []
    for i in xrange(len(feature_vector)):
        if feature_vector[i] != 0:
            sparse_vec.append((i, feature_vector[i]))
    sparse_vec = svmapi.Sparse(sparse_vec)
    return sparse_vec
    '''
    return feature_vector

def loss(y, ybar):
    return Levenshtein.distance(y, ybar)


def find_most_violated_contraint(x, y):
    # W = [w_0, w_2, ..., w_47, wt_0_0, wt_0_1, wt_0_47, wt_1_0, wt_1_1, .., wt_1_47, .. wt_47_47]
    # w_i = w[i*39:(i+1)*39], i=0..47
    # wt_i_j = w[39*48 + i*48 + j], i=0..47, j=0..47
    prob = [ [ (exp(w_i dot x0), i) for i in xrange(48) ] ]  # init prob of each state at iteration 0
    prev_state = []
    for m in xrange(1, len(x)):  # for frame x1, x2, .. x(m-1)
        # P(state s at iteraion m) = max { Prev_P(j)*P(s|j)*P(x_m|s), j=0..47}
        prob_m = []  # prob of each state at iteration m
        for s in xrange(48):
            prob_s_m_all = [ (prob[m-1][i] * exp(wt_i_s) * exp(w_s dot x_m) , i) for i in xrange(48) ]
            prob_s_m = max(prob_s_m_all)
            prob_m.append(prob_s_m)
        prob.append(prob_m)
    # backtrack based on prob[m-1]

def classify_example(x):
    pass

if __name__ == '__main__':
    #read()
    '''
    # test psi()
    x = [
        [1]*39,
        [2]*39,
        [4]*39,
        [8]*39,
    ]
    y = [0,0,5,1]
    vec = psi(x, y)
    print vec[:39]
    print vec[5*39:6*39]
    print vec[1*39:2*39]
    print vec[39*48]
    print vec[39*48:39*48+6]
    print vec[39*48+48*5:39*48+48*5+2]
    '''
    #print loss('aaBc','aacb')

