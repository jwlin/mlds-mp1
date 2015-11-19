import theano
import theano.tensor as T
import numpy as np
import sys
from itertools import izip
import time
from rnnparser import TestParser

post_file = '../dataset/posteriorgram/train3.post'
test_file = '../dataset/posteriorgram/test3.post'
label_file = '../dataset/label/train.lab'
label_map_file = '../dataset/phones/48_39.map'
chr_map_file = '../dataset/48_idx_chr.map_b'
epoch_cycle = 4
batch_size = 1
#mu =  np.float32(0.00002)
lr = theano.shared(np.float32(0.0025))
learning_rate_decay = 0.99999
TestParser.load(label_map_file, chr_map_file, label_file, post_file)

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
N_INPUT = TestParser.dimension_x
N_OUTPUT = TestParser.dimension_y

x_seq = T.matrix('input')
y_hat = T.matrix('target')
x_seq2 = T.matrix('input')


#Wi,bh,Wo,bo,Wh,Wi2,bh2 = TestParser.load_matrix(fname = "rnn_parameter.txt")
Wi = TestParser.load_matrix(N_INPUT, N_HIDDEN, name='Wi')
bh = TestParser.load_matrix(N_HIDDEN, name='bh')
Wo = TestParser.load_matrix(N_HIDDEN,N_OUTPUT, name='Wo')
bo = TestParser.load_matrix(N_OUTPUT, name='bo')
Wh = TestParser.load_matrix(N_HIDDEN,N_HIDDEN, name='Wh')

Wi2 = TestParser.load_matrix(N_HIDDEN, N_HIDDEN, name='Wi2')
bh2 = TestParser.load_matrix(N_HIDDEN, name='bh2')

parameters = [Wi,bh,Wo,bo,Wh,Wi2,bh2]

def softmax(w):
    e = T.exp(np.array(w)/1.0)
    dist = e / T.sum(e)
    return dist


def sigmoid(z):
    return 1/(1+T.exp(-z))

def ReLU(z):
	return T.switch(z>0,z,0)

def step(x_t,x_t2,a_tm1,a2_tm1,y_tm1,a_tm2,a2_tm2,y_tm2):

    a_t = sigmoid( T.dot(x_t,Wi) + T.dot(a_tm1,Wh) + bh )
    a_t2 = sigmoid( T.dot(x_t2,Wi) + T.dot(a_tm2,Wh) + bh )
    
    a2_t = sigmoid( T.dot(a_t,Wi2) + T.dot(a2_tm1,Wh) + bh2 )
    a2_t2 = sigmoid( T.dot(a_t2,Wi2) + T.dot(a2_tm2,Wh) + bh2 )
    #a_t = ReLU( T.dot(x_t,Wi) + T.dot(a_tm1,Wh) + bh )
    y_t = softmax(T.dot(a2_t,Wo) + bo)
    y_t2 = softmax(T.dot(a2_t2,Wo) + bo)
    #y_t = T.dot(a_t,Wo) + bo
    return a_t,a2_t,y_t,a_t2,a2_t2,y_t2

a_0 = theano.shared(np.zeros(N_HIDDEN).astype(np.float32))
a2_0 = theano.shared(np.zeros(N_HIDDEN).astype(np.float32))
y_0 = theano.shared(np.zeros(N_OUTPUT).astype(np.float32))
a_1 = theano.shared(np.zeros(N_HIDDEN).astype(np.float32))
a2_1 = theano.shared(np.zeros(N_HIDDEN).astype(np.float32))
y_1 = theano.shared(np.zeros(N_OUTPUT).astype(np.float32))

[a_seq,a2_seq,y_seq,a_seq2,a2_seq2,y_seq2],_ = theano.scan(
    step,
    sequences =[x_seq, x_seq2],
    outputs_info = [ a_0,a2_0,y_0, a_1,a2_1,y_1 ],
    truncate_gradient=-1
)
y_seq_all = ( y_seq + y_seq2[::-1] ) / 2.0
#cost = T.sum( ( y_seq - y_hat )**2 ) 
cost = T.nlinalg.trace(-T.log(T.dot( y_seq_all , y_hat.T )))
gradients = T.grad(cost,parameters)


    
def MyUpdate(parameters,gradients):
  
    lr_updates = [(lr,lr*learning_rate_decay)]
    parameters_updates = [(p,p - lr * g) for p,g in izip(parameters,gradients) ] 
    return parameters_updates+lr_updates

rnn_test = theano.function(
    inputs= [x_seq,x_seq2],
    outputs=y_seq_all,
    allow_input_downcast=True
)

rnn_train = theano.function(
    inputs=[x_seq,y_hat,x_seq2],
    outputs=cost,
    updates=MyUpdate(parameters,gradients),
    allow_input_downcast=True
)




t_start = time.time()

for i in xrange(epoch_cycle):
    for j in xrange(len(TestParser.x_seq)):
        c_cost = rnn_train(TestParser.x_seq[j], TestParser.y_hat[j] ,TestParser.x_seq[j][::-1])/len(TestParser.y_hat[j])
        print"iteration:", j, "cost:",  c_cost
    '''
    batch_x,batch_y = TestParser.batch(batch_size)
    for j in xrange(len(batch_x)):
        c_cost = rnn_train(batch_x[j], batch_y[j],batch_x[j][::-1])/len( batch_y[j])
        print"iteration:", j, "cost:",  c_cost'''
    

TestParser.save_parameters(parameters)
t_end = time.time()
print 'time elapsed: %f minutes' % ((t_end-t_start)/60.0)

result, test_id, ans = TestParser.load_test_data(post_file)
t_start = time.time()
correct = 0
count = 0
max_index = 0

for i in xrange(len(result)):
    frame = test_id[i]
    y_a = rnn_test(result[test_id[i]],result[test_id[i]][::-1])
    y_a = list(y_a)
    for j in xrange(len(y_a)):
        y_a[j] = list(y_a[j])
        
        max_index = y_a[j].index(max(y_a[j]))
        if TestParser.label[ans[test_id[i]][j]] == TestParser.label_keys[max_index]: correct+=1
        count+=1
        '''frame_id = ans[frame][j]
        with open('train3.post','a') as cf:
            cf.write(str(frame_id))
            for k in xrange(len(y_a[j])):
                cf.write(' '+str(y_a[j][k]))
            cf.write('\n')   ''' 
        
t_end = time.time()
print "correct = %d / %d, %f mins" %(correct, count, (t_end-t_start)/60.0)


result, test_id, ans = TestParser.load_test_data(test_file)
max_index = 0
solution_hw1 = []
solution_hw2 = []
tmp_hw1=[]
for i in xrange(len(result)):
    frame = test_id[i]
    phone_seq = ''
    
    y_a = rnn_test(result[frame],result[frame][::-1])
    y_a = list(y_a)
    tmp=[]
    for j in xrange(len(y_a)):
        y_a[j] = list(y_a[j])
        frame_id = ans[frame][j]
        max_index = y_a[j].index(max(y_a[j]))
        predicted_label = TestParser.label_keys[max_index]
        hw1_label = TestParser.label_map[predicted_label]
        solution_hw1.append( (frame_id, hw1_label) )
        tmp.append(hw1_label)
        phone_seq += TestParser.chr_map[hw1_label][1]
        '''with open('test3.post','a') as cf:
            cf.write(str(frame_id))
            for k in xrange(len(y_a[j])):
                cf.write(' '+str(y_a[j][k]))
            cf.write('\n')  '''
    phone_seq = TestParser.trim(phone_seq)
    solution_hw2.append( (frame, phone_seq) )
    tmp_hw1.append(tmp)



for i in xrange(len(tmp_hw1)):
    tmp_hw1[i][0]='sil'
    tmp_hw1[i][1]='sil'
    tmp_hw1[i][2]='sil'
    tmp_hw1[i][len(tmp_hw1[i])-1]='sil'
    tmp_hw1[i][len(tmp_hw1[i])-2]='sil'
    last = tmp_hw1[i][1]
    for j  in xrange(3, len(tmp_hw1[i])-2):
        cur = tmp_hw1[i][j]
        if cur!=last:
            if cur==tmp_hw1[i][j+1] and cur==tmp_hw1[i][j+2]:
                last = cur
            else: #tmp_hw1[i][j]=''
                tmp_hw1[i][j]=last

tmp_count=0
with open('solution_hw1_trim.csv', 'w') as f:
    f.write('Id,Prediction\n')
    for i in xrange(len(tmp_hw1)):
        for j in xrange(len(tmp_hw1[i])):
            f.write(solution_hw1[tmp_count][0] + ',' + tmp_hw1[i][j] + '\n')
            tmp_count+=1

with open('solution_hw2_trim.csv', 'w') as f:
    f.write('id,phone_sequence\n')
    for i in xrange(len(tmp_hw1)):
        frame = test_id[i]
        tmp_seq=''
        for j in xrange(len(tmp_hw1[i])):
            if tmp_hw1[i][j] == '': continue
            tmp_seq += TestParser.chr_map[ tmp_hw1[i][j] ][1]
        tmp_seq = TestParser.trim(tmp_seq)
        f.write(frame + ',' + tmp_seq + '\n')



with open('solution_hw1.csv', 'w') as f:
    f.write('Id,Prediction\n')
    for i in xrange(len(solution_hw1)):
        f.write(solution_hw1[i][0] + ',' + solution_hw1[i][1] + '\n')
with open('solution_hw2.csv', 'w') as f:
    f.write('id,phone_sequence\n')
    for i in xrange(len(solution_hw2)):
        f.write(solution_hw2[i][0] + ',' + solution_hw2[i][1] + '\n')

