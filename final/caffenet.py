import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './python')
import caffe
import time
import os

start_time = time.time()

caffe_root = './'

caffe.set_mode_cpu()
'''
net = caffe.Net(caffe_root + 'models/bvlc_googlenet/deploy.prototxt',
                caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel',
                caffe.TEST)
'''
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

with open(caffe_root + 'caffenet_top1.txt', 'w') as f:
    f.write('img_id\tdescription\n')
with open(caffe_root + 'caffenet_top5.txt', 'w') as f:
    f.write('img_id\tdescription\n')

img_dir = ['../mlds-final/train2014', '../mlds-final/val2014']
for d in img_dir:
    for imgname in os.listdir(d):
        image_id = imgname[-10:-4]
        image_path = os.path.join(d, imgname)
        #net.blobs['data'].reshape(1,3,224,224)
        net.blobs['data'].reshape(1,3,227,227)
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image_path))
        out = net.forward()
        #print("Predicted class is #{}.".format(out['prob'][0].argmax()))
        # sort top k predictions from softmax output
        top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
        top_1 = '"' + labels[top_k[0]][10:] + '"'
        top_5 = ''
        for index in top_k:
            top_5 += labels[index][10:] + " "
        top_5 = '"' + top_5 + '"'

        with open(caffe_root + 'caffenet_top1.txt', 'a') as f:
            f.write(image_id + '\t' + top_1 + '\n')
        with open(caffe_root + 'caffenet_top5.txt', 'a') as f:
            f.write(image_id + '\t' + top_5 + '\n')

        #plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
        #plt.show()
        #raw_input()

print time.time() - start_time, 'secs'
