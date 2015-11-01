import theano
import theano.tensor as T
import numpy as np
import sys
from itertools import izip
import time
from rnnparser import TestParser


# Number of units in the hidden (recurrent) layer
N_HIDDEN = 32
# input
N_INPUT = 48
# output
N_OUTPUT = 48



# what can we get from gen_data()
#print gen_data()

x_seq = T.matrix('input')
y_hat = T.matrix('target')

Wi = theano.shared( np.random.randn(N_INPUT,N_HIDDEN).astype(np.float32))
bh = theano.shared( np.zeros(N_HIDDEN).astype(np.float32) )
Wo = theano.shared( np.random.randn(N_HIDDEN,N_OUTPUT).astype(np.float32) )
bo = theano.shared( np.zeros(N_OUTPUT) .astype(np.float32) )
Wh = theano.shared( np.random.randn(N_HIDDEN,N_HIDDEN).astype(np.float32) )
parameters = [Wi,bh,Wo,bo,Wh]

def sigmoid(z):
        return 1/(1+T.exp(-z))

def step(x_t,a_tm1,y_tm1):
        a_t = sigmoid( T.dot(x_t,Wi) \
                + T.dot(a_tm1,Wh) + bh )
        y_t = T.dot(a_t,Wo) + bo
        return a_t, y_t

a_0 = theano.shared(np.zeros(N_HIDDEN).astype(np.float32))
y_0 = theano.shared(np.zeros(N_OUTPUT).astype(np.float32))

[a_seq,y_seq],_ = theano.scan(
                        step,
                        sequences = x_seq,
                        outputs_info = [ a_0, y_0 ],
                        truncate_gradient=-1
                )

 # we only care about the last output 
cost = T.sum( ( y_seq - y_hat )**2 ) 

gradients = T.grad(cost,parameters)

def MyUpdate(parameters,gradients):
    mu =  np.float32(0.0001)
    parameters_updates = [(p,p - mu * g) for p,g in izip(parameters,gradients) ] 
    return parameters_updates

rnn_test = theano.function(
        inputs= [x_seq],
        outputs=y_seq,
        allow_input_downcast=True
)

rnn_train = theano.function(
        inputs=[x_seq,y_hat],
        outputs=cost,
        updates=MyUpdate(parameters,gradients),
        allow_input_downcast=True

)

x_seq, y_hat = TestParser.load()

for i in xrange(1):
    for j in xrange(len(x_seq)):
        c_cost = rnn_train(x_seq[j],y_hat[j])/len(y_hat[j])
        print"iteration:", j, "cost:",  c_cost
        
t_start = time.time()
correct = 0
all = 0
max_index = 0
for i in xrange(len(x_seq)):
    y_a = rnn_test(x_seq[i])
    y_a = list(y_a)
    for j in xrange(len(y_a)):
        y_a[j] = list(y_a[j])
        max_index = y_a[j].index(max(y_a[j]))
        if TestParser.chr_keys[max_index] == TestParser.label_ans[all]: correct+=1
        all+=1
t_end = time.time()
print "correct = ",correct ,"/",all ,t_end-t_start," sec." 