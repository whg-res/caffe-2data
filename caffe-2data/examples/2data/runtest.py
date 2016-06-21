import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
#%matplotlib inline
import scipy.misc
from PIL import Image
import scipy.io
import os


# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/hed/
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

param = sys.argv

# load test data
data_root = ''
with open('test_test.txt') as f:
    test_lst = f.readlines()
    
test_lst = [data_root+x.strip() for x in test_lst]

idx = int(param[1])
idx_name = ""
im_lst = []
for i in range(0, len(test_lst)):
    im = Image.open(test_lst[i])
    if i==idx:
        im.save("original.jpg", "JPEG")
	idx_name = test_lst[i]
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    im_lst.append(in_)

in_ = im_lst[idx]
in_ = in_.transpose((2,0,1))

#remove the following two lines if testing with cpu
caffe.set_mode_gpu()
caffe.set_device(0)

# load net
model_root = './'
net = caffe.Net(model_root+'deploy.prototxt', model_root+'trained_model.caffemodel', caffe.TEST)

# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_

# run net and take argmax for prediction
net.forward()

# visualize the result
rgbArray = np.zeros((32,32,3), 'uint8')
rgbArray[..., 0] = net.blobs['reconst2'].data[0][2,:,:]
rgbArray[..., 1] = net.blobs['reconst2'].data[0][1,:,:]
rgbArray[..., 2] = net.blobs['reconst2'].data[0][0,:,:]
print idx_name
result_img = Image.fromarray(rgbArray)
result_img.save("result.jpg", "JPEG")
