from __future__ import division, print_function
path = 'data/dogscats/'

import os, json
from glob import glob
import numpy as np 
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt 

import utils; reload(utils)
from utils import plots 

batch_size = 64

import vgg16; reload(vgg16)
from vgg16 import Vgg16
vgg = Vgg16()
batches = vgg.get_batches(path+'train', batch_size=batch_size)
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)