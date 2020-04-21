import pdb
import argparse
import os
import time
import logging
from random import uniform
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from ast import literal_eval
from torch.nn.utils import clip_grad_norm
from math import ceil
import numpy as np
import scipy.optimize as sciopt
import warnings
from sklearn import random_projection as rp
import re
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)
import torch
torch.manual_seed(1337)
from torch.autograd import Variable
import torch.nn.functional as F
import keras #This dependency is only for loading the CIFAR-10 data set
from keras.datasets import cifar10
from copy import deepcopy
import vgg

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_train = np.transpose(X_train, axes=(0, 3, 1, 2))
X_test = X_test.astype('float32')
X_test = np.transpose(X_train, axes=(0, 3, 1, 2))
X_train /= 255
X_test /= 255

model = vgg.vgg11_bn()


# Forward pass
opfun = lambda X: model.forward(Variable(torch.from_numpy(X)))

# Forward pass through the network given the input
predsfun = lambda op: np.argmax(op.data.numpy(), 1)

# Do the forward pass, then compute the accuracy
accfun   = lambda op, y: np.mean(np.equal(predsfun(op), y.squeeze()))*100

# Initial point
x0 = deepcopy(model.state_dict())

# Number of epochs to train for
# Choose a large value since LB training needs higher values
# Changed from 150 to 30
nb_epochs = 30


# If SB.pth and LB.pth are available
# set hotstart = True and run only the
# parametric plot (i.e., don't train the network)
hotstart = False

if not hotstart:
    for fractions_of_dataset in [200]: #Run with 1/10th the data set and 1/200th the dataset
        optimizer = torch.optim.Adam(model.parameters())
        model.load_state_dict(x0)
        average_loss_over_epoch = '-'
        print('Optimizing the network with batch size %d' % (X_train.shape[0]/fractions_of_dataset))
        np.random.seed(1337) #So that both networks see same sequence of batches
        for e in range(2):
            model.eval()
            print('Epoch:', e, ' of ', nb_epochs, 'Average loss:', average_loss_over_epoch)
            average_loss_over_epoch = 0.
            # Checkpoint the model every epoch
            torch.save(model.state_dict(), 'LB.pth')

            # Training loop!
            for smpl in np.split(np.random.permutation(range(X_train.shape[0])), fractions_of_dataset):
                model.train()
                optimizer.zero_grad()
                ops = opfun(X_train[smpl])
                tgts = Variable(torch.from_numpy(y_train[smpl]).long().squeeze())
                loss_fn = F.nll_loss(ops, tgts)
                average_loss_over_epoch += loss_fn.data.numpy() / fractions_of_dataset
                loss_fn.backward()
                optimizer.step()

# Load stored values
# If hotstarted, loop is ignored and SB/LB files must be provided
mbatch = torch.load('LB.pth')

print(mbatch.iteritems())
