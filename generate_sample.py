import random

sample = {}
label = {}
num = 100000
with open('fbank/train.ark', 'r') as f:
	for line in f:
		#line = line.replace('\n', '')
		token = line.split(' ')
		sample_id = token[0]
		attr = ' '.join(token[1:])
		sample[sample_id] = attr
with open('label/train.lab', 'r') as f:
	for line in f:
		#line = line.replace('\n', '')
		token = line.split(',')
		sample_id = token[0]			
		label[sample_id] = ',' + token[1]

keys = list(sample.keys())
random.shuffle(keys)
with open('fbank/train-3.ark', 'a') as f1:
	with open('label/train-3.lab', 'a') as f2:
		for i in xrange(num):
			sample_id = keys.pop()
			f1.write(sample_id + ' ' + sample[sample_id])
			f2.write(sample_id + label[sample_id])

