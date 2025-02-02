import theano
import theano.tensor as T
import numpy as np
import sys
from itertools import izip
import time
from rnnparser import TestParser

post_file = '../dataset/posteriorgram/train2.post'
test_file = '../dataset/posteriorgram/test.post'
label_file = '../dataset/label/train.lab'
label_map_file = '../dataset/phones/48_39.map'
chr_map_file = '../dataset/48_idx_chr.map_b'
epoch_cycle = 10
mu =  np.float32(0.005)
TestParser.load(label_map_file, chr_map_file, label_file, post_file)

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 64
N_INPUT = TestParser.dimension_x
N_OUTPUT = TestParser.dimension_y

x_seq = T.matrix('input')
y_hat = T.matrix('target')

#Wi,bh,Wo,bo,Wh = TestParser.load_matrix(fname = "rnn_parameter.txt")
Wi = TestParser.load_matrix(N_INPUT, N_HIDDEN, name='Wi')
bh = TestParser.load_matrix(N_HIDDEN, name='bh')
Wo = TestParser.load_matrix(N_HIDDEN,N_OUTPUT, name='Wo')
bo = TestParser.load_matrix(N_OUTPUT, name='bo')
Wh = TestParser.load_matrix(N_HIDDEN,N_HIDDEN, name='Wh')
parameters = [Wi,bh,Wo,bo,Wh]

def softmax(w):
    e = T.exp(np.array(w)/1.0)
    dist = e / T.sum(e)
    return dist


def sigmoid(z):
    return 1/(1+T.exp(-z))

def ReLU(z):
	return T.switch(z>0,z,0)

def step(x_t,a_tm1,y_tm1):
    a_t = sigmoid( T.dot(x_t,Wi) + T.dot(a_tm1,Wh) + bh )
    #a_t = ReLU( T.dot(x_t,Wi) + T.dot(a_tm1,Wh) + bh )
    y_t = softmax(T.dot(a_t,Wo) + bo)
    #y_t = T.dot(a_t,Wo) + bo
    return a_t, y_t

a_0 = theano.shared(np.zeros(N_HIDDEN).astype(np.float32))
y_0 = theano.shared(np.zeros(N_OUTPUT).astype(np.float32))

[a_seq,y_seq],_ = theano.scan(
    step,
    sequences = x_seq,
    outputs_info = [ a_0, y_0 ],
    truncate_gradient=-1
)

y_seq_last = y_seq # we only care about the last output 
#cost = T.sum( ( y_seq_last - y_hat )**2 ) 
cost = T.nlinalg.trace(-T.log(T.dot( y_seq_last , y_hat.T )))
gradients = T.grad(cost,parameters)



    
def MyUpdate(parameters,gradients):
    parameters_updates = [(p,p - mu * g) for p,g in izip(parameters,gradients) ] 
    return parameters_updates

rnn_test = theano.function(
    inputs= [x_seq],
    outputs=y_seq_last,
    allow_input_downcast=True
)

rnn_train = theano.function(
    inputs=[x_seq,y_hat],
    outputs=cost,
    updates=MyUpdate(parameters,gradients),
    allow_input_downcast=True
)
t_start = time.time()
for i in xrange(epoch_cycle):
    for j in xrange(len(TestParser.x_seq)):
        c_cost = rnn_train(TestParser.x_seq[j], TestParser.y_hat[j])/len(TestParser.y_hat[j])
        print"iteration:", j, "cost:",  c_cost
       
    #print(i,'th epoch')
TestParser.save_parameters(parameters)
t_end = time.time()
print 'time elapsed: %f minutes' % ((t_end-t_start)/60.0)

result, test_id, ans = TestParser.load_test_data(post_file)
t_start = time.time()
correct = 0
count = 0
max_index = 0

for i in xrange(len(result)):
    y_a = rnn_test(result[test_id[i]])
    #y_a = result[test_id[i]]
    y_a = list(y_a)
    for j in xrange(len(y_a)):
        y_a[j] = list(y_a[j])
        max_index = y_a[j].index(max(y_a[j]))
       
        if TestParser.label[ans[test_id[i]][j]] == TestParser.label_keys[max_index]: correct+=1
        count+=1
t_end = time.time()
print "correct = %d / %d, %f mins" %(correct, count, (t_end-t_start)/60.0)
