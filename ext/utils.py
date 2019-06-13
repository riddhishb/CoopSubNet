##########################################################################
#                                                                         #
# Desc : All additional functionalities for cooperative networks         #
#         and their experiments, like, data handling/processing, shuffle, #
#         etc.                                                              #
# Author : Riddhish Bhalodia                                             #
# Institution : Scientific Computing and Imaging Institute                 #
# Date : 18th December 2018                                                 #
#                                                                         #
##########################################################################

from __future__ import division
import numpy as np
import numpy.matlib 
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import csv
import os
import sys
from urllib.request import urlretrieve
import tarfile
import pickle
import zipfile
import nrrd
import xml.etree.ElementTree as ET
from scipy.io import loadmat
import tfmpl
import scipy as sp
import subprocess

def readSpecificTagXML(root, tagName):
    '''
    Reading a specific tag from 
    '''
    d = root.find(tagName)
    d = d.text
    return d.strip('\n')

def modifyXML(basexml, tagName, newTag):
    tree = ET.parse(basexml)
    root = tree.getroot()
    d = root.find(tagName)
    d.text = str(newTag)
    tree.write(basexml)

def parametric_relu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.01))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

def conv3d(x, W):
    """3d convolution with full stride"""
    return tf.nn.conv3d(x, W, strides = [1,1,1,1,1], padding='SAME')

def conv2d(x, W):
    """2d convolution with full stride"""
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')

def max_pool_2x2_3d(x):
    """max pool downsamples by 2"""
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1],padding='SAME')

def max_pool_3x3_3d(x):
    """max pool downsamples by 3"""
    return tf.nn.max_pool3d(x, ksize=[1,3,3,3,1], strides=[1,3,3,3,1],padding='SAME')

def max_pool_2x2_2d(x):
    """max pool downsamples by 2"""
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

def max_pool_3x3_2d(x):
    """max pool downsamples by 3"""
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1],padding='SAME')

def weight_variable(shape, name):
    """weight tf variable of given shape"""
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=True))

def bias_variable(shape, name):
    """bias tf variable of given shape"""
    init = tf.constant_initializer(0.0)
    return tf.get_variable(name=name, shape=shape , initializer=init)

def batch_norm(x, shape, phase_train, reuse):
    beta = tf.Variable(tf.constant(0.0, shape=shape), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=shape), name='gamma', trainable=True)

    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')


    with tf.name_scope(None):
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        # with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def parametric_relu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def conv_layer(input_d, indim, outdim, name, ks, phase_train, bn=True, reuse=False):
    '''
    Defining convolution layer with/without batch normalization
    '''
    with tf.name_scope(name):
        nmwt = name + '_wt'
        nmbias = name + '_bias'
        cv = weight_variable([ks, ks, indim, outdim], nmwt)
        if bn:
            hc = conv2d(input_d, cv)
            hcbn = batch_norm(hc, [outdim], phase_train, reuse)
            out = parametric_relu(hcbn)
        else:
            biasc = bias_variable([outdim], nmbias)
            hcb = conv2d(input_d, cv) + biasc
            out = parametric_relu(hcb)
        return out

def conv_layer_reg(input_d, indim, outdim, name, ks, phase_train, bn=True, reuse=False):
    '''
    Defining convolution layer with/without batch normalization
    '''
    with tf.name_scope(name):
        nmwt = name + '_wt'
        nmbias = name + '_bias'
        cv = weight_variable([ks, ks, indim, outdim], nmwt)
        if bn:
            hc = conv2d(input_d, cv)
            hcbn = batch_norm(hc, [outdim], phase_train, reuse)
            out = parametric_relu(hcbn)
        else:
            biasc = bias_variable([outdim], nmbias)
            hcb = conv2d(input_d, cv) + biasc
            out = parametric_relu(hcb)
        return [out, tf.nn.l2_loss(cv)]

def reduceMNIST(mnist, redF):
    train_data = mnist.train.images
    train_labels = mnist.train.labels
    out_train_data = np.zeros([redF*10,784])
    out_train_labels = np.zeros([redF*10,10])

    for i in range(10):
        f = np.where(train_labels[:, i] == 1)
        f = f[0]
        np.random.shuffle(f)
        idx = f[:redF]
        out_train_data[i*redF:(i+1)*redF, :] = train_data[idx, :]
        out_train_labels[i*redF:(i+1)*redF, :] = train_labels[idx, :]

    t = np.arange(redF*10)
    np.random.shuffle(t)
    out_train_data = out_train_data[t, :]
    out_train_labels = out_train_labels[t, :]
    return [out_train_data, out_train_labels]

def reduceCIFAR(data, labels, redF):
    nLab = labels.shape[1]
    out_train_data = np.zeros([redF*nLab, data.shape[1]])
    out_train_labels = np.zeros([redF*nLab, nLab])
    out_val_data = np.zeros([500*nLab, data.shape[1]])
    out_val_labels = np.zeros([500*nLab, nLab])

    for i in range(nLab):
        f = np.where(labels[:, i] == 1)
        f = f[0]
        np.random.shuffle(f)
        idxTr = f[:redF]
        idxVal = f[redF+1:redF+501]
        out_train_data[i*redF:(i+1)*redF, :] = data[idxTr, :]
        out_train_labels[i*redF:(i+1)*redF, :] = labels[idxTr, :]
        out_val_data[i*500:(i+1)*500, :] = data[idxVal, :]
        out_val_labels[i*500:(i+1)*500, :] = labels[idxVal, :]

    tr = np.arange(redF*nLab)
    vl = np.arange(500*nLab)
    np.random.shuffle(tr)
    np.random.shuffle(vl)
    out_train_data = out_train_data[tr, :]
    out_train_labels = out_train_labels[tr, :]
    out_val_data = out_val_data[vl, :]
    out_val_labels = out_val_labels[vl, :]
    return [out_train_data, out_train_labels, out_val_data, out_val_labels]

# @tfmpl.figure_tensor
# def draw_scatter(scaled, colors): 
#     '''Draw scatter plots. One for each color.'''  
#     figs = tfmpl.create_figures(len(colors), figsize=(4,4))
#     for idx, f in enumerate(figs):
#         ax = f.add_subplot(111)
#         ax.axis('off')
#         ax.scatter(scaled[:, 0], scaled[:, 1], c=colors[idx])
#         f.tight_layout()

#     return figs

# def variable_summaries(var):
#     """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean', mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
# 	    tf.summary.scalar('stddev', stddev)
# 	    tf.summary.scalar('max', tf.reduce_max(var))
# 	    tf.summary.scalar('min', tf.reduce_min(var))
# 	    tf.summary.histogram('histogram', var)

def get_batch_data(data_tensor, data_labels, idx, batch_size):
    batch_tensor = data_tensor[idx*batch_size : (idx+1)*batch_size, ...]
    batch_labels = data_labels[idx*batch_size : (idx+1)*batch_size, ...]
    return [batch_tensor, batch_labels]

def training_shuffle_data(origData, origLabels):
    idx = np.arange(origData.shape[0])
    shufIdx = np.copy(idx)
    np.random.shuffle(shufIdx)
    shufData = origData[shufIdx, :]
    if origLabels.ndim == 1:
        origLabels = origLabels.reshape(len(origLabels), 1)
    shufLabels = origLabels[shufIdx, :]
    return [shufData, shufLabels]

def training_shuffle_data_SS(origData, origLabels, origMasks):
  lab_idx = np.where(origMasks == 1)[0]
  np.random.shuffle(lab_idx)
  lab_data = origData[lab_idx, ...]
  lab_labels = origLabels[lab_idx, ...]
  unlab_idx = np.where(origMasks == 0)[0]
  np.random.shuffle(unlab_idx)
  unlab_data = origData[unlab_idx, :]
  unlab_labels = origLabels[unlab_idx, :]
  return [lab_data, lab_labels, unlab_data, unlab_labels]


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def maybe_download_and_extract():
        main_directory = "data_set/"
        cifar_10_directory = main_directory+"cifar_10/"
        if not os.path.exists(main_directory):
                os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
                zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
                tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory+"./cifar-10-batches-py", cifar_10_directory)
        os.remove(zip_cifar_10)

def get_data_set(name="train"):
        x = None
        y = None

        # uncomment this when you need to download data
        # maybe_download_and_extract()

        folder_name = "cifar_10"

        f = open('./data_set/'+folder_name+'/batches.meta', 'rb')
        f.close()

        if name is "train":
                for i in range(5):
                        f = open('./data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
                        datadict = pickle.load(f, encoding='latin1')
                        f.close()

                        _X = datadict["data"]
                        _Y = datadict['labels']

                        _X = np.array(_X, dtype=float) / 255.0
                        _X = _X.reshape([-1, 3, 32, 32])
                        _X = _X.transpose([0, 2, 3, 1])
                        _X = _X.reshape(-1, 32*32*3)

                        if x is None:
                                x = _X
                                y = _Y
                        else:
                                x = np.concatenate((x, _X), axis=0)
                                y = np.concatenate((y, _Y), axis=0)

        elif name is "test":
                f = open('./data_set/'+folder_name+'/test_batch', 'rb')
                datadict = pickle.load(f, encoding='latin1')
                f.close()

                x = datadict["data"]
                y = np.array(datadict['labels'])

                x = np.array(x, dtype=float) / 255.0
                x = x.reshape([-1, 3, 32, 32])
                x = x.transpose([0, 2, 3, 1])
                x = x.reshape(-1, 32*32*3)

        return x, dense_to_one_hot(y)


def dense_to_one_hot(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

def _print_download_progress(count, block_size, total_size):
        pct_complete = float(count * block_size) / total_size
        msg = "\r- Download progress: {0:.1%}".format(pct_complete)
        sys.stdout.write(msg)
        sys.stdout.flush()

def limitCIFAR(data, labels, redL):
    sizeNew = 0
    for i in range(redL):
        f = np.where(labels[:, i] == 1)
        f = f[0]
        sizeNew += len(f)
    newData = np.zeros([sizeNew, data.shape[1]])
    newLabels = np.zeros([sizeNew, redL])
    sizeNew = 0
    for i in range(redL):
        f = np.where(labels[:, i] == 1)
        f = f[0]
        newData[sizeNew:sizeNew+len(f), :] = data[f, :]
        newLabels[sizeNew:sizeNew+len(f), :] = labels[f, :redL]
        sizeNew += len(f)
    return [newData, newLabels]

class landmarkDataPartition:
  def __init__(self, imgPaths, ptPaths, imgSize, numL):
    self.imgPathsAll=[]
    with open(imgPaths, 'r') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=',')
      for row in spamreader:
        self.imgPathsAll.append(row)

    self.imgData = np.zeros([len(self.imgPathsAll), imgSize[0]*imgSize[1]*3])
    for i in range(len(self.imgPathsAll)):
      imgname = self.imgPathsAll[i][0]
      # print(imgname[0])
      img = plt.imread(imgname)
      self.imgData[i, ...] = img.reshape([1, imgSize[0]*imgSize[1]*3])

    self.ptPathsAll=[]
    with open(ptPaths, 'r') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=',')
      for row in spamreader:
        self.ptPathsAll.append(row)

    self.ptData = np.zeros([len(self.imgPathsAll), numL*2])
    for i in range(len(self.imgPathsAll)):
      pt = np.loadtxt(self.ptPathsAll[i][0])
      pt[..., 0] = 640*pt[...,0]
      pt[..., 1] = 480*pt[...,1]
      self.ptData[i, ...] = pt.reshape(1, numL*2)

    totNum = len(self.imgPathsAll)
    idx = np.arange(totNum)
    np.random.shuffle(idx)
    traEndpt = int(np.floor(0.7*totNum))
    traValpt = int(np.floor(0.9*totNum))
    self.idxTrain = idx[:traEndpt]
    self.idxVal = idx[traEndpt + 1:traValpt]
    self.idxTest = idx[traValpt + 1:]

  def trainImages(self, ld):
    if ld == 0:
      trImg = self.imgData[self.idxTrain, ...]
      np.save('/home/sci/riddhishb/Documents/git-repos/cooperativenetworks/data/FaceLandmarks/trainImages.npy', trImg)
    else:
      print("Alternate")
      trImg = np.load('/home/sci/riddhishb/Documents/git-repos/cooperativenetworks/data/FaceLandmarks/trainImages.npy')
    return trImg
  def valImages(self, ld):
    if ld == 0:
      valImg = self.imgData[self.idxVal, ...]
      np.save('/home/sci/riddhishb/Documents/git-repos/cooperativenetworks/data/FaceLandmarks/valImages.npy', valImg)
    else:
      print("Alternate")
      valImg = np.load('/home/sci/riddhishb/Documents/git-repos/cooperativenetworks/data/FaceLandmarks/valImages.npy')
    return valImg
  def testImages(self, ld):
    if ld == 0:
      testImg = self.imgData[self.idxTest, ...]
      np.save('/home/sci/riddhishb/Documents/git-repos/cooperativenetworks/data/FaceLandmarks/testImages.npy', testImg)
    else:
      print("Alternate")
      testImg = np.load('/home/sci/riddhishb/Documents/git-repos/cooperativenetworks/data/FaceLandmarks/testImages.npy')
    return testImg
  def trainPoints(self, ld):
    if ld == 0:
      trPt = self.ptData[self.idxTrain, ...]
      np.save('/home/sci/riddhishb/Documents/git-repos/cooperativenetworks/data/FaceLandmarks/trainPoints.npy', trPt)
    else:
      print("Alternate")
      trPt = np.load('/home/sci/riddhishb/Documents/git-repos/cooperativenetworks/data/FaceLandmarks/trainPoints.npy')
    return trPt
  def valPoints(self, ld):
    if ld == 0:
      valPt = self.ptData[self.idxVal, ...]
      np.save('/home/sci/riddhishb/Documents/git-repos/cooperativenetworks/data/FaceLandmarks/valPoints.npy', valPt)
    else:
      print("Alternate")
      valPt = np.load('/home/sci/riddhishb/Documents/git-repos/cooperativenetworks/data/FaceLandmarks/valPoints.npy')
    return valPt
  def testPoints(self, ld):
    if ld == 0:
      testPt = self.ptData[self.idxTest, ...]
      np.save('/home/sci/riddhishb/Documents/git-repos/cooperativenetworks/data/FaceLandmarks/testPoints.npy', testPt)
    else:
      print("Alternate")
      testPt = np.load('/home/sci/riddhishb/Documents/git-repos/cooperativenetworks/data/FaceLandmarks/testPoints.npy')
    return testPt

def partitionMNIST_SS(mnist, redF):
  train_data = mnist.train.images
  train_labels = mnist.train.labels
  mask = np.zeros([train_data.shape[0], 1])
  for i in range(10):
    f = np.where(train_labels[:, i] == 1)
    f = f[0]
    np.random.shuffle(f)
    idx = f[:redF]
    mask[idx, 0] = 1
  return [train_data, train_labels, mask]

def partitionToy_SS(data, labels, redF):
    d = data.shape[1]
    valData = np.zeros([400, d])
    valLabels = np.zeros([400, 4])
    testData = np.zeros([400, d])
    testLabels = np.zeros([400, 4])
    trainData = np.zeros([1200, d])
    trainLabels = np.zeros([1200, 4])
    mask = np.zeros([trainData.shape[0], 1])
    for i in range(4):
        valData[(i)*100:(i+1)*100, ...] = data[(i)*500:(i)*500+100, ...]
        testData[(i)*100:(i+1)*100, ...] = data[(i)*500+100:(i)*500+200, ...]
        trainData[(i)*300:(i+1)*300, ...] = data[(i)*500+200:(i)*500+500, ...]
        valLabels[(i)*100:(i+1)*100, ...] = labels[(i)*500:(i)*500+100, ...]
        testLabels[(i)*100:(i+1)*100, ...] = labels[(i)*500+100:(i)*500+200, ...]
        trainLabels[(i)*300:(i+1)*300, ...] = labels[(i)*500+200:(i)*500+500, ...]

    for i in range(4):
        f = np.where(trainLabels[:, i] == 1)
        f = f[0]
        # np.random.shuffle(f)
        idx = f[:redF]
        mask[idx, 0] = 1

    return [trainData, trainLabels, mask, valData, valLabels, testData, testLabels]

def partitionToy(data, labels, redF):
    d = data.shape[1]
    valData = np.zeros([400, d])
    valLabels = np.zeros([400, 4])
    testData = np.zeros([400, d])
    testLabels = np.zeros([400, 4])
    trainData = np.zeros([1200, d])
    trainLabels = np.zeros([1200, 4])
    trainDataAct = np.zeros([4*redF, d])
    trainLabelsAct = np.zeros([4*redF, 4])
    for i in range(4):
        valData[(i)*100:(i+1)*100, ...] = data[(i)*500:(i)*500+100, ...]
        testData[(i)*100:(i+1)*100, ...] = data[(i)*500+100:(i)*500+200, ...]
        trainData[(i)*300:(i+1)*300, ...] = data[(i)*500+200:(i)*500+500, ...]
        valLabels[(i)*100:(i+1)*100, ...] = labels[(i)*500:(i)*500+100, ...]
        testLabels[(i)*100:(i+1)*100, ...] = labels[(i)*500+100:(i)*500+200, ...]
        trainLabels[(i)*300:(i+1)*300, ...] = labels[(i)*500+200:(i)*500+500, ...]

    for i in range(4):
        f = np.where(trainLabels[:, i] == 1)
        f = f[0]
        # np.random.shuffle(f)
        idx = f[:redF]
        # mask[idx, 0] = 1
        trainDataAct[i*redF:(i+1)*redF, ...] = trainData[idx, ...]
        trainLabelsAct[i*redF:(i+1)*redF, ...] = trainLabels[idx, ...]

    return [trainDataAct, trainLabelsAct, valData, valLabels, testData, testLabels]

def partitionToyReg(data, labels, redF):
    d = data.shape[1]
    valData = np.zeros([400, d])
    valLabels = np.zeros([400, 1])
    testData = np.zeros([400, d])
    testLabels = np.zeros([400, 1])
    trainData = np.zeros([1200, d])
    trainLabels = np.zeros([1200, 1])
    trainDataAct = np.zeros([redF, d])
    trainLabelsAct = np.zeros([redF, 1])
    valData = data[:400, ...]
    testData = data[400:800, ...]
    trainData = data[800:1200, ...]

    valLabels = labels[:400, ...]
    testLabels = labels[400:800, ...]
    trainLabels = labels[800:1200, ...]
    idx = np.arange(800, 1200)
    # np.random.shuffle(idx)
    trainDataAct = trainData[:redF, ...]
    trainLabelsAct = trainLabels[:redF, ...]
    return [trainDataAct, trainLabelsAct, valData, valLabels, testData, testLabels]

def partitionToy(data, labels, redF):
    d = data.shape[1]
    valData = np.zeros([400, d])
    valLabels = np.zeros([400, 4])
    testData = np.zeros([400, d])
    testLabels = np.zeros([400, 4])
    trainData = np.zeros([1200, d])
    trainLabels = np.zeros([1200, 4])
    trainDataAct = np.zeros([4*redF, d])
    trainLabelsAct = np.zeros([4*redF, 4])
    for i in range(4):
        valData[(i)*100:(i+1)*100, ...] = data[(i)*500:(i)*500+100, ...]
        testData[(i)*100:(i+1)*100, ...] = data[(i)*500+100:(i)*500+200, ...]
        trainData[(i)*300:(i+1)*300, ...] = data[(i)*500+200:(i)*500+500, ...]
        valLabels[(i)*100:(i+1)*100, ...] = labels[(i)*500:(i)*500+100, ...]
        testLabels[(i)*100:(i+1)*100, ...] = labels[(i)*500+100:(i)*500+200, ...]
        trainLabels[(i)*300:(i+1)*300, ...] = labels[(i)*500+200:(i)*500+500, ...]

    for i in range(4):
        f = np.where(trainLabels[:, i] == 1)
        f = f[0]
        # np.random.shuffle(f)
        idx = f[:redF]
        # mask[idx, 0] = 1
        trainDataAct[i*redF:(i+1)*redF, ...] = trainData[idx, ...]
        trainLabelsAct[i*redF:(i+1)*redF, ...] = trainLabels[idx, ...]

    return [trainDataAct, trainLabelsAct, valData, valLabels, testData, testLabels]

def processNuclie(trainImagesPath, trainPointsPath, testImagesPath, testPointsPaths):
  imgPathsTrain=[]
  with open(trainImagesPath, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
      imgPathsTrain.append(row)

  ptPathsTrain=[]
  with open(trainPointsPath, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
      ptPathsTrain.append(row)

  imgPathsTest=[]
  with open(testImagesPath, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
      imgPathsTest.append(row)

  ptPathsTest=[]
  with open(testPointsPaths, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
      ptPathsTest.append(row)

  timg = plt.imread(imgPathsTrain[0][0])
  sz = timg.size
  szpt = int(sz/3.0)
  out_trainImg = np.zeros([len(imgPathsTrain), sz])
  out_testImg = np.zeros([len(imgPathsTest), sz])
  out_trainPt = np.zeros([len(ptPathsTrain), szpt])
  out_testPt = np.zeros([len(ptPathsTest), szpt])
  for i in range(len(imgPathsTrain)):
    # print(i)
    imgnm = imgPathsTrain[i][0]
    img = plt.imread(imgnm)
    out_trainImg[i, ...] = img.reshape(1, sz)
    imgpt = np.zeros([img.shape[0], img.shape[1]], dtype=np.float32)
    ptnm = ptPathsTrain[i][0]
    pt = loadmat(ptnm)
    pts = np.floor(pt['detection']).astype(np.int16)
    w = np.where(pts >= 500)[0]
    pts[w, ...] = pts[w, ...] - 1
    # print(pts)
    imgpt[pts[..., 1], pts[..., 0]] = 1.0
    imgpt = sp.ndimage.morphology.binary_dilation(imgpt, iterations=3)
    imgpt = imgpt.astype(np.float32)
    imgpt = sp.ndimage.gaussian_filter(imgpt, 0.4)
    out_trainPt[i, ...] = imgpt.reshape(1, szpt)

  for i in range(len(imgPathsTest)):
    imgnm = imgPathsTest[i][0]
    img = plt.imread(imgnm)
    out_testImg[i, ...] = img.reshape(1, sz)
    imgpt = np.zeros([img.shape[0], img.shape[1]])
    ptnm = ptPathsTest[i][0]
    pt = loadmat(ptnm)
    pts = np.floor(pt['detection']).astype(np.int16)
    w = np.where(pts >= 500)[0]
    pts[w, ...] = pts[w, ...] - 1
    imgpt[pts[..., 1], pts[..., 0]] = 1.0
    imgpt = sp.ndimage.morphology.binary_dilation(imgpt, iterations=3)
    imgpt = imgpt.astype(np.float32)
    imgpt = sp.ndimage.gaussian_filter(imgpt, 0.4)
    out_testPt[i, ...] = imgpt.reshape(1, szpt)

  return [out_trainImg, out_trainPt, out_testImg, out_testPt]

def makepatches(imgData, ptData, pathcsz, stride, imgSize):
  initNum = imgData.shape[0]
  tmp1 = (imgSize[0] - pathcsz[0]) / stride[0] + 1
  tmp2 = (imgSize[1] - pathcsz[1]) / stride[1] + 1
  finalNum = int(initNum * tmp1 * tmp2)
  newimgData = np.zeros([finalNum, pathcsz[0]*pathcsz[1]*3])
  newptData = np.zeros([finalNum, pathcsz[0]*pathcsz[1]])

  ## make the patches
  count = 0
  for i in range(initNum):
    img = imgData[i, ...].reshape(imgSize[0], imgSize[1], 3)
    imgPt = ptData[i, ...].reshape(imgSize[0], imgSize[1])
    xPatchStart = np.linspace(0, imgSize[0] - pathcsz[0], tmp1).astype(np.int16)
    yPatchStart = np.linspace(0, imgSize[1] - pathcsz[1], tmp2).astype(np.int16)
    for x in xPatchStart:
      for y in yPatchStart:
        ptch = img[x:x + pathcsz[0], y: y + pathcsz[1], :]
        ptchpt = imgPt[x:x + pathcsz[0], y: y + pathcsz[1]]
        newimgData[count, ...] = ptch.reshape(1, pathcsz[0]*pathcsz[1]*3)
        newptData[count, ...] = ptchpt.reshape(1, pathcsz[0]*pathcsz[1])
        count += 1

  return [newimgData, newptData]
################## RESNET Building Blocks ###############################

def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
                                             data_format):
    """A single block for ResNet v1, without a bottleneck.
    Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the convolutions.
        training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
        projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
        strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training,
                                                    data_format=data_format)

    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=1,
            data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                                                 strides, data_format):
    """A single block for ResNet v1, with a bottleneck.
    Similar to _building_block_v1(), except using the "bottleneck" blocks
    described in:
        Convolution then batch normalization then ReLU as described by:
            Deep Residual Learning for Image Recognition
            https://arxiv.org/pdf/1512.03385.pdf
            by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the convolutions.
        training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
        projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
        strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training,
                                                    data_format=data_format)

    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=1,
            data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
            data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs

# def projection_shortcut(inputs):
#     return conv2d_fixed_padding(
#             inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
#             data_format=data_format)

#     # Only the first block per block_layer uses projection_shortcut and strides
#     inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
#                                         data_format)

#     for _ in range(1, blocks):
#         inputs = block_fn(inputs, filters, training, None, 1, data_format)

#     return tf.identity(inputs, name)
################## RESNET Building Blocks ###############################