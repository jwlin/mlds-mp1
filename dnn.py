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

sample_file = 'mfcc/train.ark'
test_file = 'mfcc/test.ark'
label_file = 'label/train.lab'
label_map_file = 'phones/48_39.map'
DataParser.load(sample_file, label_file, label_map_file)
#DataParser.test()
dim_x = DataParser.dimension_x
dim_y_hat = DataParser.dimension_y
batch_size = 21
neuron_num = 64
epoch_cycle = 2
learning_rate = 0.0001
learning_rate_decay = 0.9999

# e.g. matrix 3*2 dot matrix 2*1 = matrix 3*1
# [[1., 3.], [2., 2.], [3.,1.]] dot [[2], [1]] = [[5.], [6.], [7.]]
x = T.matrix('input', dtype='float64')  # matrix of dim_x * batch_size
y_hat = T.matrix('reference', dtype='float64')  # matrix of dim_y_hat * batch_size

# load saved parameters
#w1,w2,w3,b1,b2,b3 = DataParser.load_matrix(fname = 'parameter.txt')

# if no saved parameters, generate them randomly
w1 = DataParser.load_matrix(neuron_num, dim_x, name='w1')  # matrix of neurom_num * dim_x
b1 = DataParser.load_matrix(neuron_num, name='b1')  # matrix of neuron_num * 1
w2 = DataParser.load_matrix(neuron_num, neuron_num, name='w2')  # matrix of neuron_num * neuron_num
b2 = DataParser.load_matrix(neuron_num, name = 'b2')  # matrix of neuron_num * 1
w3 = DataParser.load_matrix(dim_y_hat, neuron_num , name = 'w3')  # matrix of matrix of dim_y_hat * neuron_num
b3 = DataParser.load_matrix(dim_y_hat, name = 'b3')  # matrix of dim_y_hat * 1

z1 = T.dot(w1, x) + b1.dimshuffle(0, 'x')
a1 = activation_func(z1, 'relu')
z2 = T.dot(w2, a1) + b2.dimshuffle(0, 'x')
a2 = activation_func(z2, 'relu')
z3 = T.dot(w3, a2) + b3.dimshuffle(0, 'x')
y = activation_func(z3, 'relu')

parameters = [w1, w2, w3, b1, b2, b3]
cost = cost_func(y, y_hat, batch_size, 'euclidean')
gradients = T.grad(cost, parameters)

def my_update(params, gradients):
	global learning_rate
	global learning_rate_decay
	learning_rate *= learning_rate_decay
	param_updates = [
		(p, p-learning_rate*g) for p, g in izip(params, gradients)
	]
	return param_updates

train = theano.function(
		inputs = [x, y_hat],
		outputs = cost,
		updates = my_update(parameters, gradients),
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

DataParser.save_parameters(parameters)

t_end = time.time()
print 'time elapsed: %f minutes' % ((t_end-t_start)/60.0)
