"""
Reproduces the parametric plot experiment from the paper
for a network like C3.

Plots a parametric plot between SB and LB
minimizers demonstrating the relative sharpness
of the two minima.

Requirements:
- Keras (only for CIFAR-10 dataset; easy to avoid)
- Matplotlib
- Numpy

TODO:
- Enable the code to run on CUDA as well.
  (As of now, it only runs on CPU)

Run Command:
        python plot_parametric_pytorch.py

The plot is saved as C3ish.pdf
"""

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
X_test = np.transpose(X_test, axes=(0, 3, 1, 2))
X_train /= 255
X_test /= 255

# This is where you can load any model of your choice.
# I stole PyTorch Vision's VGG network and modified it to work on CIFAR-10.
# You can take this line out and add any other network and the code
# should run just fine.
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
nb_epochs = 2


# If SB.pth and LB.pth are available
# set hotstart = True and run only the
# parametric plot (i.e., don't train the network)
hotstart = False
#
# if not hotstart:
#     for fractions_of_dataset in [100, 200]: #Run with 1/10th the data set, until 1/2000th the dataset
#         optimizer = torch.optim.Adam(model.parameters())
#         model.load_state_dict(x0)
#         average_loss_over_epoch = '-'
#         print('Optimizing the network with batch size %d' % (X_train.shape[0]/fractions_of_dataset))
#         np.random.seed(1337) #So that both networks see same sequence of batches
#         for e in range(nb_epochs):
#             model.eval()
#             print('Epoch:', e, ' of ', nb_epochs, 'Average loss:', average_loss_over_epoch)
#             average_loss_over_epoch = 0
#             # Checkpoint the model every epoch
#             torch.save(model.state_dict(), "BatchSize" + str(X_train.shape[0]//fractions_of_dataset) + ".pth")
#
#             # Training loop!
#             for smpl in np.split(np.random.permutation(range(X_train.shape[0])), fractions_of_dataset):
#                 model.train()
#                 optimizer.zero_grad()
#                 ops = opfun(X_train[smpl])
#                 tgts = Variable(torch.from_numpy(y_train[smpl]).long().squeeze())
#                 loss_fn = F.nll_loss(ops, tgts)
#                 average_loss_over_epoch += loss_fn.data.numpy() / fractions_of_dataset
#                 loss_fn.backward()
#                 optimizer.step()


print('Loaded stored solutions')

#Functions relevant for calculating Sharpness

def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    if 0 and len(0) > 1:
        model = torch.nn.DataParallel(model, 0)
    # print(data_loader)
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    grad_vec = None
    if training:
      optimizer = torch.optim.SGD(model.parameters(), 1.0)
      optimizer.zero_grad()  # only zerout at the beginning


    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if 0 is not None:
            continue# target = target.cuda(device=None) #commented out
        input_var = Variable(inputs.type(torch.cuda.FloatTensor), volatile=not training)
        target_var = Variable(target)

        # compute output
        if not training:
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
            losses.update(loss.data[0], input_var.size(0))
            top1.update(prec1[0], input_var.size(0))
            top5.update(prec5[0], input_var.size(0))

        else:
            mini_inputs = input_var.chunk(256 // 256)
            mini_targets = target_var.chunk(256 // 256)

            for k, mini_input_var in enumerate(mini_inputs):

                mini_target_var = mini_targets[k]
                output = model(mini_input_var)
                loss = criterion(output, mini_target_var)

                prec1, prec5 = accuracy(output.data, mini_target_var.data, topk=(1, 5))
                losses.update(loss.data[0], mini_input_var.size(0))
                top1.update(prec1[0], mini_input_var.size(0))
                top5.update(prec5[0], mini_input_var.size(0))

                # compute gradient and do SGD step
                loss.backward()

            #optimizer.step() # no step in this case

    # reshape and averaging gradients
    if training:
        for p in model.parameters():
            if p.grad is not None: # new line
                p.grad.data.div_(len(data_loader))
                if grad_vec is None:
                    grad_vec = p.grad.data.view(-1)
                else:
                    grad_vec = torch.cat((grad_vec, p.grad.data.view(-1)))

    #logging.info('{phase} - \t'
    #             'Loss {loss.avg:.4f}\t'
    #             'Prec@1 {top1.avg:.3f}\t'
    #             'Prec@5 {top5.avg:.3f}'.format(
    #              phase='TRAINING' if training else 'EVALUATING',
    #              loss=losses, top1=top1, top5=top5))

    return {'loss': losses.avg,
            'prec1': top1.avg,
            'prec5': top5.avg}, grad_vec


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    raise NotImplementedError('train functionality is changed. Do not use it!')


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    res, _ = forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)
    return res


def get_minus_cross_entropy(x, data_loader, model, criterion, training=False):
  if type(x).__module__ == np.__name__:
    x = torch.from_numpy(x).float()
    # x = x.cuda()
  # switch to evaluate mode
  model.eval()

  # fill vector x of parameters to model
  x_start = 0
  for p in model.parameters():
    psize = p.data.size()
    peltnum = 1
    for s in psize:
      peltnum *= s
    x_part = x[x_start:x_start+peltnum]
    p.data = x_part.view(psize)
    x_start += peltnum

  result, grads = forward(data_loader, model, criterion, 0,
                 training=training, optimizer=None)
  #print ('get_minus_cross_entropy {}!'.format(-result['loss']))
  return (-result['loss'], None if grads is None else grads.cpu().numpy().astype(np.float64))

def get_sharpness(data_loader, model, criterion, epsilon, manifolds=0):

  # extract current x0
  x0 = None
  for p in model.parameters():
    if x0 is None:
      x0 = p.data.view(-1)
    else:
      x0 = torch.cat((x0, p.data.view(-1)))
  x0 = x0.cpu().numpy()

  # get current f_x
  f_x0, _ = get_minus_cross_entropy(x0, data_loader, model, criterion)
  f_x0 = -f_x0
  logging.info('min loss f_x0 = {loss:.4f}'.format(loss=f_x0))

  # find the minimum
  if 0==manifolds:
    x_min = np.reshape(x0 - epsilon * (np.abs(x0) + 1), (x0.shape[0], 1))
    x_max = np.reshape(x0 + epsilon * (np.abs(x0) + 1), (x0.shape[0], 1))
    bounds = np.concatenate([x_min, x_max], 1)
    func = lambda x: get_minus_cross_entropy(x, data_loader, model, criterion, training=True)
    init_guess = x0
  else:
    warnings.warn("Small manifolds may not be able to explore the space.")
    assert(manifolds<=x0.shape[0])
    #transformer = rp.GaussianRandomProjection(n_components=manifolds)
    #transformer.fit(np.random.rand(manifolds, x0.shape[0]))
    #A_plus = transformer.components_
    #A = np.linalg.pinv(A_plus)
    A_plus = np.random.rand(manifolds, x0.shape[0])*2.-1.
    # normalize each column to unit length
    A_plus_norm = np.linalg.norm(A_plus, axis=1)
    A_plus = A_plus / np.reshape(A_plus_norm, (manifolds,1))
    A = np.linalg.pinv(A_plus)
    abs_bound = epsilon * (np.abs(np.dot(A_plus, x0))+1)
    abs_bound = np.reshape(abs_bound, (abs_bound.shape[0], 1))
    bounds = np.concatenate([-abs_bound, abs_bound], 1)
    def func(y):
      floss, fg = get_minus_cross_entropy(x0 + np.dot(A, y), data_loader, model, criterion, training=True)
      return floss, np.dot(np.transpose(A), fg)
    #func = lambda y: get_minus_cross_entropy(x0+np.dot(A, y), data_loader, model, criterion, training=True)
    init_guess = np.zeros(manifolds)

  #rand_selections = (np.random.rand(bounds.shape[0])+1e-6)*0.99
  #init_guess = np.multiply(1.-rand_selections, bounds[:,0])+np.multiply(rand_selections, bounds[:,1])

  print(init_guess.shape)
  for i in range(len(bounds)):
      bounds[i] = (bounds[i][0],bounds[i][1])
  minimum_x, f_x, d = sciopt.fmin_l_bfgs_b(func, init_guess, maxiter=10, bounds=list(bounds), disp=1)
    #factr=10.,
    #pgtol=1.e-12,

  f_x = -f_x
  logging.info('max loss f_x = {loss:.4f}'.format(loss=f_x))
  sharpness = (f_x - f_x0)/(1+f_x0)*100

  # recover the model
  x0 = torch.from_numpy(x0).float()
  x0 = x0.cuda()
  x_start = 0
  for p in model.parameters():
      psize = p.data.size()
      peltnum = 1
      for s in psize:
          peltnum *= s
      x_part = x0[x_start:x_start + peltnum]
      p.data = x_part.view(psize)
      x_start += peltnum

  return sharpness


fractions_of_dataset = [100, 200]
fractions_of_dataset.reverse()
grid_size = len(fractions_of_dataset) #How many points of interpolation between [0, 5000]
data_for_plotting = np.zeros((grid_size, 3)) #four lines  --> change to 3 in Figure 4
# batch_range = np.linspace(0, 5000, grid_size)

i = 0

#Fill in test accuracy values
#for `grid_size' points in the interpolation
# for fraction in fractions_of_dataset:
#     mydict = {}
#     batchmodel = torch.load("BatchSize" + str(X_train.shape[0]//fraction) + ".pth")
#     for key, value in batchmodel.items():
#         mydict[key] = value
#     model.load_state_dict(mydict)
#
#     j = 0
#     for datatype in [(X_train, y_train), (X_test, y_test)]:
#         dataX = datatype[0]
#         datay = datatype[1]
#         for smpl in np.split(np.random.permutation(range(dataX.shape[0])), 10):
#             ops = opfun(dataX[smpl])
#             tgts = Variable(torch.from_numpy(datay[smpl]).long().squeeze())
#             # data_for_plotting[i, j] +=
#             var = F.nll_loss(ops, tgts).data.numpy() / 10
#             if j == 1:
#                 data_for_plotting[i, j-1] += accfun(ops, datay[smpl]) / 10.
#         j += 1
#     print(data_for_plotting[i])
#     i += 1
# np.save('intermediate-values', data_for_plotting)

# Data loading code
default_transform = {
    'train': get_transform("cifar10",
                           input_size=None, augment=True),
    'eval': get_transform("cifar10",
                          input_size=None, augment=False)
}
transform = getattr(model, 'input_transform', default_transform)

# define loss function (criterion) and optimizer
criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
criterion.type(torch.cuda.FloatTensor)
# model.type(torch.cuda.FloatTensor)

# logging.info('\nValidation Loss {val_loss:.4f} \t'
#              'Validation Prec@1 {val_prec1:.3f} \t'
#              'Validation Prec@5 {val_prec5:.3f} \n'
#              .format(val_loss=val_loss,
#                      val_prec1=val_prec1,
#                      val_prec5=val_prec5))

sharpnesses1eNeg3 = []
sharpnesses5eNeg4 = []
i = 0
#for batch in bactch_range
for fraction in fractions_of_dataset:
    mydict = {}
    batchmodel = torch.load("BatchSize" + str(X_train.shape[0]//fraction) + ".pth")
    for key, value in batchmodel.items():
        mydict[key] = value
    model.load_state_dict(mydict)
    val_data = get_dataset("cifar10", 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=X_train.shape[0]//fraction, shuffle=False,
        num_workers=8, pin_memory=True) #batch

    val_result = validate(val_loader, model, criterion, 0)
    val_loss, val_prec1, val_prec5 = [val_result[r]
                                      for r in ['loss', 'prec1', 'prec5']]

    sharpness = get_sharpness(val_loader, model, criterion, 0.001, manifolds=0)
    sharpnesses1eNeg3.append(sharpness)
    data_for_plotting[i, 1] += sharpness
    sharpness = get_sharpness(val_loader, model, criterion, 0.0005, manifolds=0)
    sharpnesses5eNeg4.append(sharpness)
    data_for_plotting[i, 2] += sharpness
    i += 1


# logging.info('sharpness {} = {}'.format(time,sharpness))
# logging.info('sharpnesses = {}'.format(str(sharpnesses)))
# _std = np.std(sharpnesses)*np.sqrt(5)/np.sqrt(5-1)
# _mean = np.mean(sharpnesses)



# Actual plotting;
# if matplotlib is not available, use any tool of your choice by
# loading intermediate-values.npy
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.semilogy(batch_range, data_for_plotting[:, 0], 'b-')

ax2.plot(batch_range, data_for_plotting[:, 1], 'r-')
ax2.plot(batch_range, data_for_plotting[:, 2], 'r--')

ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Testing Accuracy', color='b')
ax2.set_ylabel('Sharpness', color='r')
ax1.legend(('1e-3', '5e-4'), loc=0)

ax1.grid(b=True, which='both')
plt.savefig('AccuracySharpnessPlot.pdf')
print('Saved figure; Task complete')
