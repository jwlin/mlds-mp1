"""A module for SVM^python for multiclass learning."""

# Thomas Finley, tfinley@gmail.com
import paths
from svmparser import TestParser
import numpy as np
import Levenshtein
import svmapi
#import theano

'''def read_examples(filename, sparm):
    """Parses an input file into an example sequence."""
    # This reads example files of the type read by SVM^multiclass.
    examples = []
    # Open the file and read each example.
    for line in file(filename):
        # Get rid of comments.
        if line.find('#'): line = line[:line.find('#')]
        tokens = line.split()s
        # If the line is empty, who cares?
        if not tokens: continue
        # Get the target.
        target = int(tokens[0])
        # Get the features.
        tokens = [t.split(':') for t in tokens[1:]]
        features = [(0,1)]+[(int(k),float(v)) for k,v in tokens]
        # Add the example to the list
        examples.append((svmapi.Sparse(features), target))
    # Print out some very useful statistics.
    print len(examples),'examples read'
    return examples'''

def init_model(sample, sm, sparm):
    """Store the number of features and classes in the model."""
    # Note that these features will be stored in the model and written
    # when it comes time to write the model to a file, and restored in
    # the classifier when reading the model from the file.
    sm.num_features = max(max(x) for x,y in sample)[0]+1
    sm.num_classes = max(y for x,y in sample)
    sm.size_psi = sm.num_features*sm.num_classes
    #print 'size_psi set to',sm.size_psi

thecount = 0
# vim: set ts=4 sw=4 et: -*- coding: utf-8 -*-


#import cPickle
#import csv

def read_examples(filename, sparm):
    print 'in example'
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
    print 'after example',len(examples)
    return examples
def psi(x, y, sm, sparm):
    print 'in psi'
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
    
    # do sparse format
    sparse_vec = []
    for i in xrange(len(feature_vector)):
        if feature_vector[i] != 0:
            sparse_vec.append((i, feature_vector[i]))
    sparse_vec = svmapi.Sparse(sparse_vec)
    return sparse_vec
    

def loss(y, ybar,sparm):
    print 'in loss'
    return Levenshtein.distance(y, ybar)


def find_most_violated_contraint(x, y,sm, sparm):
    # W = [w_0, w_2, ..., w_47, wt_0_0, wt_0_1, wt_0_47, wt_1_0, wt_1_1, .., wt_1_47, .. wt_47_47]
    # w_i = w[i*39:(i+1)*39], i=0..47
    # wt_i_j = w[39*48 + i*48 + j], i=0..47, j=0..47
    print 'in  find'
    w = sm.w
    prob = [ [ (exp(np.dot(w[i*39:(i+1)*39],x[0])), i) for i in xrange(48) ] ]  # init prob of each state at iteration 0
    prev_state = []
    
    for m in xrange(1, len(x)):  # for frame x1, x2, .. x(m-1)
        # P(state s at iteraion m) = max { Prev_P(j)*P(s|j)*P(x_m|s), j=0..47}
        prob_m = []  # prob of each state at iteration m
        for s in xrange(48):
            prob_s_m_all = [ np.log((prob[m-1][i]) +np.log( exp(w[39*48+i*48+j])) +np.log( exp(np.dot(w[s*39:(s+1)*39],x[m]))) , i) for i in xrange(48) ]
            prob_s_m = max(prob_s_m_all)
            
            prob_m.append(prob_s_m)
        prob.append(prob_m)
    # backtrack based on prob[m-1]
    ybar=[]
    max = max(prob[len(x)-1])[1]
    ybar.append(max)
    for m in xrange(len(x)-2,-1,-1):
        max = prob[m][max][1]
        ybar.append(max)
    return ybar[::-1]
       
def classify_example(x, sm, sparm):
    # W = [w_0, w_2, ..., w_47, wt_0_0, wt_0_1, wt_0_47, wt_1_0, wt_1_1, .., wt_1_47, .. wt_47_47]
    # w_i = w[i*39:(i+1)*39], i=0..47
    # wt_i_j = w[39*48 + i*48 + j], i=0..47, j=0..47
    print 'in  class'
    w = sm.w
    prob = [ [ (exp(np.dot(w[i*39:(i+1)*39],x[0])), i) for i in xrange(48) ] ]  # init prob of each state at iteration 0
    prev_state = []
    
    for m in xrange(1, len(x)):  # for frame x1, x2, .. x(m-1)
        # P(state s at iteraion m) = max { Prev_P(j)*P(s|j)*P(x_m|s), j=0..47}
        prob_m = []  # prob of each state at iteration m
        for s in xrange(48):
            prob_s_m_all = [ np.log((prob[m-1][i]) +np.log( exp(w[39*48+i*48+j])) +np.log( exp(np.dot(w[s*39:(s+1)*39],x[m]))) , i) for i in xrange(48) ]
            prob_s_m = max(prob_s_m_all)
            
            prob_m.append(prob_s_m)
        prob.append(prob_m)
    # backtrack based on prob[m-1]
    ybar=[]
    max = max(prob[len(x)-1])[1]
    ybar.append(max)
    for m in xrange(len(x)-2,-1,-1):
        max = prob[m][max][1]
        ybar.append(max)
    return ybar[::-1]

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


