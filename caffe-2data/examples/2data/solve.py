from __future__ import division
import numpy as np
import sys
caffe_root = '../../' 
sys.path.insert(0, caffe_root + 'python')
import caffe


# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')

# load weight if needed
#base_weights = 'trained_model.caffemodel'
#solver.net.copy_from(base_weights)

# start training
solver.step(100000)

