import theano
import theano.tensor as T
import random
import numpy as np
import time
from itertools import izip
from data_parser import DataParser

def activation_func(z, choice='sigmoid'):
	if choice == 'sigmoid':
		return 1/(1+T.exp(-1*z))
	elif choice == 'relu':
		return T.maximum(z, 0)
	else:
		assert False, 'no activation function'

def cost_func(y, y_hat, batch_size, choice='euclidean'):
	if choice == 'euclidean':
		return T.sum((y-y_hat)**2) / batch_size
	else:
		assert False, 'no cost function'


t_start = time.time()

sample_file = 'mfcc/train-1.ark'
test_file = 'mfcc/test.ark'
label_file = 'label/train-1.lab'
label_map_file = 'phones/48_39.map'
DataParser.load(sample_file, label_file, label_map_file)
#DataParser.test()
dim_x = DataParser.dimension_x
dim_y_hat = DataParser.dimension_y
batch_size = 10
neuron_num = 64
epoch_cycle = 2 
learning_rate = 0.1
lr = theano.shared(learning_rate)
adagrad_t = theano.shared(0)

# e.g. matrix 3*2 dot matrix 2*1 = matrix 3*1
# [[1., 3.], [2., 2.], [3.,1.]] dot [[2], [1]] = [[5.], [6.], [7.]]
x = T.matrix('input', dtype='float64')  # matrix of dim_x * batch_size
y_hat = T.matrix('reference', dtype='float64')  # matrix of dim_y_hat * batch_size

# load saved parameters
#w1,w2,w3,b1,b2,b3 = DataParser.load_matrix(fname = 'parameter.txt')

# if no saved parameters, generate them randomly
w1 = DataParser.load_matrix(neuron_num, dim_x, name='w1')  # matrix of neurom_num * dim_x
w1_g = theano.shared(np.zeros((neuron_num, dim_x)), name='w1_g')
b1 = DataParser.load_matrix(neuron_num, name='b1')  # matrix of neuron_num * 1
b1_g = theano.shared(np.zeros(neuron_num), name = 'b1_g')
w2 = DataParser.load_matrix(neuron_num, neuron_num, name='w2')  # matrix of neuron_num * neuron_num
w2_g = theano.shared(np.zeros((neuron_num, neuron_num)), name='w2_g')
b2 = DataParser.load_matrix(neuron_num, name = 'b2')  # matrix of neuron_num * 1
b2_g = theano.shared(np.zeros(neuron_num), name = 'b2_g')
w3 = DataParser.load_matrix(dim_y_hat, neuron_num , name = 'w3')  # matrix of matrix of dim_y_hat * neuron_num
w3_g = theano.shared(np.zeros((dim_y_hat, neuron_num)), name='w3_g')
b3 = DataParser.load_matrix(dim_y_hat, name = 'b3')  # matrix of dim_y_hat * 1
b3_g = theano.shared(np.zeros(dim_y_hat), name = 'b3_g')

z1 = T.dot(w1, x) + b1.dimshuffle(0, 'x')
a1 = activation_func(z1, 'sigmoid')
z2 = T.dot(w2, a1) + b2.dimshuffle(0, 'x')
a2 = activation_func(z2, 'sigmoid')
z3 = T.dot(w3, a2) + b3.dimshuffle(0, 'x')
y = activation_func(z3, 'sigmoid')

parameters = [w1, w2, w3, b1, b2, b3]
grad_hists = [w1_g, w2_g, w3_g, b1_g, b2_g, b3_g]
cost = cost_func(y, y_hat, batch_size, 'euclidean')
gradients = T.grad(cost, parameters)
adagrad_t_update = [(adagrad_t, adagrad_t+1)]
lr_update = [(lr, learning_rate/T.sqrt(adagrad_t+1))]
new_grad_hists = [ g_hist + g **2 for g_hist, g in zip(grad_hists, gradients) ]
grad_hist_update = zip(grad_hists, new_grad_hists)
sigma = [ T.sqrt((1//(adagrad_t+1))*g_hist) for g_hist in grad_hists ]
param_update = [ (p, p-((lr/s)*g)) for p, g, s in izip(parameters, gradients, sigma) ]

train = theano.function(
		inputs = [x, y_hat],
		outputs = cost,
		updates = param_update + lr_update + adagrad_t_update + grad_hist_update
		)

'''
validate = theano.function(
		inputs = [x, y_hat],
		outputs = cost,
		)
'''

test = theano.function(
		inputs = [x],
		outputs = y,
		)

for t in xrange(epoch_cycle):
	print 'epoch', t
	current_cost = 0
	#validate_cost = 0
	x_batch, y_hat_batch, batch_num = DataParser.make_batch(batch_size)
	#x_batch = [[[-2.50, -1.39], [-2.50, 0.92]], [[-2.50, -1.39], [-2.49, 0.93]], [[-2.50, -1.39], [-2.48, 0.94]]]
	#y_hat_batch = [[[0,1]], [[0,1]], [[0,1]]]
	for i in xrange(batch_num):
		c_cost = train(x_batch[i], y_hat_batch[i])
		#print 'c_cost', c_cost
		current_cost += c_cost
	current_cost /= batch_num
	print 'current_cost', current_cost
	DataParser.save_parameters(parameters, 'parameter.txt')
	DataParser.save_parameters(grad_hists, 'parameter-g.txt')
	#current_validate_cost = validate(validation_set)
	#print 'validate cost', current_validate_cost
	#if current_validate_cost < validate_cost:
	#	validate_cost = current_validate_cost
	#else:
	#	break

test_data, test_id = DataParser.load_test_data(sample_file)
result = test(test_data)
result = list(result)
result = map(list, zip(*result))  # transpose
with open('result.txt', 'w') as f:
	for i in xrange(len(result)):
		f.write(test_id[i] + ',')
		max_value = 0
		max_index = -1
		candidate = []
		candidate_value = []
		for j in xrange(len(result[i])):
			if result[i][j] > 0.8:
				candidate.append(DataParser.label_index[j])
				candidate_value.append(result[i][j])
			if result[i][j] > max_value:
			    max_value = result[i][j]
			    max_index = j
		with open('candidate.txt', 'a') as cf:
			cf.write(DataParser.label_index[max_index] + ' of ' + str(candidate) + ': ' + str(candidate_value) + '\n')
		#f.write(random.choice(candidate) + '\n')
		f.write(DataParser.label_index[max_index] + '\n')

correct, all_record = DataParser.check('result.txt', label_file)
print 'correction rate: %i/ %i' %(correct, all_record)

'''
solution_data, solution_id = DataParser.load_test_data(test_file)
solution = test(solution_data)
solution = list(solution)
solution = map(list, zip(*solution))  # transpose
with open('solution.csv', 'w') as f:
	f.write('Id,Prediction\n')
	for i in xrange(len(solution)):
		f.write(solution_id[i] + ',')
		max_value = 0
		max_index = -1
		for j in xrange(len(solution[i])):
			if solution[i][j] > max_value:
				max_value = solution[i][j]
				max_index = j
		f.write(DataParser.label_map[DataParser.label_index[max_index]] + '\n')
'''

t_end = time.time()
print 'time elapsed: %f minutes' % ((t_end-t_start)/60.0)
