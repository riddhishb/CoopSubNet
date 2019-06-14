##########################################################################
#																		 #
# Desc : All network architectures for cooperative networks 			 #
# Author : Riddhish Bhalodia											 #
# Institution : Scientific Computing and Imaging Institute				 #
# Date : 18th December 2018												 #
#																		 #
##########################################################################

import numpy as np
import tensorflow as tf
import utils as ut
import csv
import os
import xml.etree.ElementTree as ET
from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt


class MNIST_Hardcon_Classification:

	'''
	This class defines the state of the art MNIST classifier.
	With the case one variation, hard constraint bottleneck
	It's network architecture
	It's graph building function
	It's training and testing
	'''

	def __init__(self, sess, xmlfilename):

		''' Initialization of variables '''
		self.sess = sess

		tree=ET.parse(xmlfilename)
		root = tree.getroot()
		# self.separator_position = int(readSpecificTagXML(root, 'separator_position'))
		self.checkpoint_dir = ut.readSpecificTagXML(root, 'checkpoint_path')
		self.result_dir = ut.readSpecificTagXML(root, 'result_dir')
		self.log_dir = ut.readSpecificTagXML(root, 'log_dir')
		self.model_name = ut.readSpecificTagXML(root, 'model_name')
		self.learning_rate = float(ut.readSpecificTagXML(root, 'learning_rate'))
		self.batch_size = int(ut.readSpecificTagXML(root, 'batch_size'))
		self.epochs = int(ut.readSpecificTagXML(root, 'num_epochs'))
		self.print_iter = int(ut.readSpecificTagXML(root, 'print_iter'))
		self.save_iter = int(ut.readSpecificTagXML(root, 'save_iter'))
		# self.data_path = readSpecificTagXML(root, 'data_path')
		self.reduced_data = int(ut.readSpecificTagXML(root, 'reduced_data')) # number of data per label to use for training
		self.bottleneck = int(ut.readSpecificTagXML(root, 'bottleneck')) 
		# read and process the data sequence
		mnist = input_data.read_data_sets("../data/MNIST/MNIST_data/", one_hot=True)
		[self.orig_data, self.orig_labels] = ut.reduceMNIST(mnist, self.reduced_data)
		print(self.orig_data.shape)
		self.orig_data_val = mnist.validation.images
		self.orig_labels_val = mnist.validation.labels

		self.orig_data_test = mnist.test.images
		self.orig_labels_test = mnist.test.labels

		self.num_batches = int(self.orig_data.shape[0] / self.batch_size)

	def feature_extractor(self, x, phase_train, is_training=True, reuse=False):

		''' 
		The initial network which represents the feature extraction
		part of deep network

		'''
		with tf.variable_scope("feature_extractor", reuse=reuse):
			x_in = tf.reshape(x, [-1, 28, 28, 1])
			conv1_1 = ut.conv_layer(x_in, 1, 64, 'conv1_1', 3, phase_train)
			# conv1_2 = ut.conv_layer(conv1_1, 64, 128, 'conv1_2', 3, phase_train)
			with tf.name_scope('pool1'):
				h_pool1 = ut.max_pool_2x2_2d(conv1_1)
			# conv2_1 = ut.conv_layer(h_pool1, 128, 256, 'conv2_1', 3, phase_train)
			conv2_2 = ut.conv_layer(h_pool1, 64, 128, 'conv2_2', 3, phase_train)
			with tf.name_scope('pool2'):
				h_pool2 = ut.max_pool_2x2_2d(conv2_2)
			
			return h_pool2


	def classifier(self, x, phase_train, is_training=True, reuse=False):

		'''
		The end part of the deep network which takes in the extracted
		features and classifies them into digits

		'''
		with tf.variable_scope('classifier', reuse=reuse):
			with tf.name_scope('fc1'):
				W_fc1 = ut.weight_variable([7*7*128, 1024], 'fc1_wt')
				b_fc1 = ut.bias_variable([1024], 'fc1_bias')
				
				h_pool2_flat = tf.reshape(x, [-1, 7*7*128])
				h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
			
			with tf.name_scope('fc2'):
				W_fc2 = ut.weight_variable([1024, self.bottleneck], 'fc2_wt')
				b_fc2 = ut.bias_variable([self.bottleneck], 'fc2_bias')
				h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

			with tf.name_scope('fc3'):
				W_fc3 = ut.weight_variable([self.bottleneck, 1024], 'fc3_wt')
				b_fc3 = ut.bias_variable([1024], 'fc3_bias')
				h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

			with tf.name_scope('fc4'):
				W_fc4 = ut.weight_variable([1024, 10], 'fc4_wt')
				b_fc4 = ut.bias_variable([10], 'fc4_bias')
				out = tf.matmul(h_fc3, W_fc4) + b_fc4
			return out

	def build_model(self):

		'''
		Builds the graph of the network, adds in the loss functions,
		defines placeholders.

		'''
		self.input_data = tf.placeholder(tf.float32, [None, 28*28], name='orig_data')
		self.labels = tf.placeholder(tf.float32, [None, 10], name='orig_labels')
		self.phase_train = tf.placeholder(tf.bool, name='phase')

		self.out_feature = self.feature_extractor(self.input_data, self.phase_train, is_training=True, reuse=False)
		self.out_labels = self.classifier(self.out_feature, self.phase_train, is_training=True, reuse=False)

		# define the cross entropy loss
		# self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(self.out_labels), reduction_indices=[1]))
		self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_labels, labels=self.labels))
		correct_pred = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.out_labels, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

		'''Training function'''
		t_vars = tf.trainable_variables()
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optim = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cross_entropy, var_list=t_vars)

	def train_model(self):

		''' 
		Implements the training routineself.

		'''
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		start_epoch = 0
		start_batch_id = 0
		counter = 1
		start_time = time.time()
		for epoch in range(start_epoch, self.epochs):
			[shufData, shufLables] = ut.training_shuffle_data(self.orig_data, self.orig_labels)
			total_accuracy_train = 0
			total_accuracy_val = 0

			for idx in range(start_batch_id, self.num_batches):
				[batch_data_train, batch_labels_train] = ut.get_batch_data(shufData, shufLables, idx, self.batch_size)
				[batch_data_val, batch_labels_val] = ut.get_batch_data(self.orig_data_val, self.orig_labels_val, idx, self.batch_size)
				self.optim.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True} )

				counter += 1

				if np.mod(counter, self.print_iter) == 0:
					# training accuracy
					accuracy_train, out_train = self.sess.run([self.accuracy, self.out_labels],
						feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:False})

					# validation accuracy
					accuracy_val, out_val = self.sess.run([self.accuracy, self.out_labels],
						feed_dict={self.input_data: batch_data_val, self.labels: batch_labels_val, self.phase_train:False})

					print("TRAINING : Epoch: [%2d] [%4d/%4d] time: %4.4f, accuracy: %.8f" %(epoch, idx, self.num_batches, time.time() - start_time, accuracy_train))
					print("VALIDATION : Epoch: [%2d] [%4d/%4d] time: %4.4f, accuracy: %.8f" %(epoch, idx, self.num_batches, time.time() - start_time, accuracy_val))

			if np.mod(epoch, self.save_iter) == 0:
				# save the check point
				self.save(self.checkpoint_dir, self.model_dir, self.model_name, counter)

	def test_model(self):

		'''
		Implements the testing routine.

		'''

		self.saver = tf.train.Saver()
		self.load(self.checkpoint_dir, self.model_dir, self.model_name)
		loadBT = 10
		numE = int(self.orig_data_test.shape[0]/loadBT)
		allaccuracy = 0
		for i in range(numE):
			batchTestData = self.orig_data_test[i*loadBT:(i+1)*loadBT, :]
			batchTestLabels = self.orig_labels_test[i*loadBT:(i+1)*loadBT, :]

			accuracy_test, out_test = self.sess.run([self.accuracy, self.out_labels],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
			allaccuracy += accuracy_test

		allaccuracy = allaccuracy / numE
		print("TESTING ACCURACY: %.8f" % (allaccuracy))


	@property
	def model_dir(self):
		return "{}".format(
			self.model_name)


	def save(self, checkpoint_dir, model_dir, model_name, step):
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir, model_name)
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name + '.model'), global_step=step)


	def load(self, checkpoint_dir, model_dir, model_name):
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir, model_name)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		print('Loading Checkpoint ...')
		print(ckpt_name)
		self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))


class MNIST_CAE_Classification:

	'''
	This class defines the state of the art MNIST classifier.
	With the case one variation, hard constraint bottleneck
	It's network architecture
	It's graph building function
	It's training and testing
	'''

	def __init__(self, sess, xmlfilename):

		''' Initialization of variables '''
		self.sess = sess

		tree=ET.parse(xmlfilename)
		root = tree.getroot()
		# self.separator_position = int(readSpecificTagXML(root, 'separator_position'))
		self.checkpoint_dir = ut.readSpecificTagXML(root, 'checkpoint_path')
		self.result_dir = ut.readSpecificTagXML(root, 'result_dir')
		self.log_dir = ut.readSpecificTagXML(root, 'log_dir')
		self.model_name = ut.readSpecificTagXML(root, 'model_name')
		self.learning_rate = float(ut.readSpecificTagXML(root, 'learning_rate'))
		self.batch_size = int(ut.readSpecificTagXML(root, 'batch_size'))
		self.epochs = int(ut.readSpecificTagXML(root, 'num_epochs'))
		self.print_iter = int(ut.readSpecificTagXML(root, 'print_iter'))
		self.save_iter = int(ut.readSpecificTagXML(root, 'save_iter'))
		# self.data_path = readSpecificTagXML(root, 'data_path')
		self.reduced_data = int(ut.readSpecificTagXML(root, 'reduced_data')) # number of data per label to use for training
		self.bottleneck = int(ut.readSpecificTagXML(root, 'bottleneck'))
		print('BOTTLENECK = ', self.bottleneck)
		self.cae_weight = float(ut.readSpecificTagXML(root, 'cae_weight')) 
		self.burn_in_epochs = int(ut.readSpecificTagXML(root, 'burn_in'))
		# read and process the data sequence
		mnist = input_data.read_data_sets("../data/MNIST/MNIST_data/", one_hot=True)
		[self.orig_data, self.orig_labels] = ut.reduceMNIST(mnist, self.reduced_data)
		# np.save('MNISTData_Limited.npy', self.orig_data)
		# np.save('MNISTLabels_Limited.npy', self.orig_labels)
		print(self.orig_data.shape)
		self.orig_data_val = mnist.validation.images
		self.orig_labels_val = mnist.validation.labels
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		self.logFN = self.log_dir + '/logFile.csv'
		self.orig_data_test = mnist.test.images
		self.orig_labels_test = mnist.test.labels

		self.num_batches = int(self.orig_data.shape[0] / self.batch_size)

	def feature_extractor(self, x, phase_train, is_training=True, reuse=False):

		''' 
		The initial network which represents the feature extraction
		part of deep network

		'''
		with tf.variable_scope("feature_extractor", reuse=reuse):
			x_in = tf.reshape(x, [-1, 28, 28, 1])
			conv1_1 = ut.conv_layer(x_in, 1, 64, 'conv1_1', 3, phase_train)
			# conv1_2 = ut.conv_layer(conv1_1, 64, 128, 'conv1_2', 3, phase_train)
			with tf.name_scope('pool1'):
				h_pool1 = ut.max_pool_2x2_2d(conv1_1)
			# conv2_1 = ut.conv_layer(h_pool1, 128, 256, 'conv2_1', 3, phase_train)
			conv2_2 = ut.conv_layer(h_pool1, 64, 128, 'conv2_2', 3, phase_train)
			with tf.name_scope('pool2'):
				h_pool2 = ut.max_pool_2x2_2d(conv2_2)
				h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*128])
			
			return h_pool2_flat


	def classifier(self, x, phase_train, is_training=True, reuse=False):

		'''
		The end part of the deep network which takes in the extracted
		features and classifies them into digits

		'''
		with tf.variable_scope('classifier', reuse=reuse):
			with tf.name_scope('fc1'):
				W_fc1 = ut.weight_variable([7*7*128, 1024], 'fc1_wt')
				b_fc1 = ut.bias_variable([1024], 'fc1_bias')	
				h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

			with tf.name_scope('fc4'):
				W_fc4 = ut.weight_variable([1024, 10], 'fc4_wt')
				b_fc4 = ut.bias_variable([10], 'fc4_bias')
				out = tf.matmul(h_fc1, W_fc4) + b_fc4
			
			return out

	def CAE(self, x, phase_train, is_training=True, reuse=False):

		'''
		The external autoencoder based regualrization
		This is the Cooperative Autoencoder (CAE)
		'''

		with tf.variable_scope('CAE', reuse=reuse):
			with tf.name_scope('fc1-cae'):
				W_fc1_cae = ut.weight_variable([7*7*128, 1024], 'fc1_wt_cae')
				b_fc1_cae = ut.bias_variable([1024], 'fc1_bias_cae')
				h_fc1_cae = ut.parametric_relu(tf.matmul(x, W_fc1_cae) + b_fc1_cae)
			with tf.name_scope('fc1b-cae'):
				W_fc1b_cae = ut.weight_variable([1024, 512], 'fc1b_wt_cae')
				b_fc1b_cae = ut.bias_variable([512], 'fc1b_bias_cae')
				h_fc1b_cae = ut.parametric_relu(tf.matmul(h_fc1_cae, W_fc1b_cae) + b_fc1b_cae)
			with tf.name_scope('fc2-cae'):
				W_fc2_cae = ut.weight_variable([512, self.bottleneck], 'fc2_wt_cae')
				b_fc2_cae = ut.bias_variable([self.bottleneck], 'fc2_bias_cae')
				h_fc2_cae = ut.parametric_relu(tf.matmul(h_fc1b_cae, W_fc2_cae) + b_fc2_cae)
			with tf.name_scope('fc3-cae'):
				W_fc3_cae = ut.weight_variable([self.bottleneck, 512], 'fc3_wt_cae')
				b_fc3_cae = ut.bias_variable([512], 'fc3_bias_cae')
				h_fc3_cae = ut.parametric_relu(tf.matmul(h_fc2_cae, W_fc3_cae) + b_fc3_cae)
			with tf.name_scope('fc3b-cae'):
				W_fc3b_cae = ut.weight_variable([512, 1024], 'fc3b_wt_cae')
				b_fc3b_cae = ut.bias_variable([1024], 'fc3b_bias_cae')
				h_fc3b_cae = ut.parametric_relu(tf.matmul(h_fc3_cae, W_fc3b_cae) + b_fc3b_cae)
			with tf.name_scope('fc4-cae'):
				W_fc4_cae = ut.weight_variable([1024, 7*7*128], 'fc4_wt_cae')
				b_fc4_cae = ut.bias_variable([7*7*128], 'fc4_bias_cae')
				h_fc4_cae = ut.parametric_relu(tf.matmul(h_fc3b_cae, W_fc4_cae) + b_fc4_cae)

			return [h_fc4_cae, h_fc2_cae]

	def build_model(self):

		'''
		Builds the graph of the network, adds in the loss functions,
		defines placeholders.

		'''
		self.input_data = tf.placeholder(tf.float32, [None, 28*28], name='orig_data')
		self.labels = tf.placeholder(tf.float32, [None, 10], name='orig_labels')
		self.phase_train = tf.placeholder(tf.bool, name='phase')
		self.out_feature = self.feature_extractor(self.input_data, self.phase_train, is_training=True, reuse=False)
		self.out_labels = self.classifier(self.out_feature, self.phase_train, is_training=True, reuse=False)
		[self.cae_out, self.bottleneck_value] = self.CAE(self.out_feature, self.phase_train, is_training=True, reuse=False)
		self.out_labels_fromCAE = self.classifier(self.cae_out, False, is_training=False, reuse=True)
		self.cae_loss = tf.reduce_mean(tf.nn.l2_loss(self.cae_out - self.out_feature)/ tf.nn.l2_loss(self.out_feature))
		self.featurenorm = tf.reduce_mean(tf.nn.l2_loss(self.out_feature))
		# define the cross entropy loss
		# self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(self.out_labels), reduction_indices=[1]))
		self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_labels, labels=self.labels))
		correct_pred = tf.equal(tf.argmax(self.labels, 1), tf.argmax(tf.nn.softmax(self.out_labels), 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
		correct_pred_CAE = tf.equal(tf.argmax(self.labels, 1), tf.argmax(tf.nn.softmax(self.out_labels_fromCAE), 1))
		self.accuracy_CAE = tf.reduce_mean(tf.cast(correct_pred_CAE, tf.float32), name='accuracy-CAE')

		self.loss = self.cross_entropy + self.cae_weight*self.cae_loss

		'''Training function'''
		t_vars = tf.trainable_variables()
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optim_burnin = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cross_entropy, var_list=t_vars)
			self.optim = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss, var_list=t_vars)
			# self.optimall = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=t_vars)

	def train_model(self):

		''' 
		Implements the training routineself.

		'''
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		all_train_loss = []
		all_train_acc = []
		all_val_loss = []
		cae_val_loss = []
		cae_train_loss = []
		match_val_loss = []
		match_train_loss = []
		all_val_acc = []
		start_epoch = 0
		start_batch_id = 0
		counter = 1
		start_time = time.time()
		with open(self.logFN, 'w') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',')
			for epoch in range(start_epoch, self.epochs):
				[shufData, shufLables] = ut.training_shuffle_data(self.orig_data, self.orig_labels)
				for idx in range(start_batch_id, self.num_batches):
					[batch_data_train, batch_labels_train] = ut.get_batch_data(shufData, shufLables, idx, self.batch_size)
					[batch_data_val, batch_labels_val] = ut.get_batch_data(self.orig_data_val, self.orig_labels_val, idx, self.batch_size)
					if epoch > self.burn_in_epochs:
						self.optim.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True} )
					else:
						self.optim_burnin.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True} )

					counter += 1
					# training accuracy

					if np.mod(counter, self.print_iter) == 0:
						# validation accuracy
						accuracy_train, match_train, cae_train, featureNT, tot_loss = self.sess.run([self.accuracy, self.cross_entropy, self.cae_loss, self.featurenorm, self.loss],
							feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:False})

						accuracy_val, match_val, cae_val, featureNV, tot_val = self.sess.run([self.accuracy, self.cross_entropy, self.cae_loss, self.featurenorm, self.loss],
							feed_dict={self.input_data: batch_data_val, self.labels: batch_labels_val, self.phase_train:False})	
						l = [epoch, idx, accuracy_train, accuracy_val, match_train, match_val, cae_train, cae_val, featureNT, featureNV]
						spamwriter.writerow(l)
						print("TRAINING : Epoch: [%2d] [%4d/%4d] time: %4.4f, accuracy: %.8f, Match-Loss: %.8f, CAE-Loss: %.8f, feature-norm: %.8f" %(epoch, idx, self.num_batches, time.time() - start_time, accuracy_train, match_train, cae_train, featureNT))
						print("VALIDATION : Epoch: [%2d] [%4d/%4d] time: %4.4f, accuracy: %.8f, Match-Loss: %.8f, CAE-Loss: %.8f, feature-norm: %.8f" %(epoch, idx, self.num_batches, time.time() - start_time, accuracy_val, match_val, cae_val, featureNV))

				if np.mod(epoch, self.save_iter) == 0:
					# save the check point
					self.save_separate(self.checkpoint_dir, self.model_dir, self.model_name, self.bottleneck, self.reduced_data, counter)

	def test_model(self):

		'''
		Implements the testing routine.
		Also computes the CAE loss for reconstruction accuracy

		'''

		self.saver = tf.train.Saver()
		self.load_separate(self.checkpoint_dir, self.model_dir, self.model_name, self.bottleneck, self.reduced_data)
		loadBT = 10
		numE = int(self.orig_data_test.shape[0]/loadBT)
		allaccuracy = 0
		allaccuracyCAE = 0
		cae_err = 0
		for i in range(numE):
			batchTestData = self.orig_data_test[i*loadBT:(i+1)*loadBT, :]
			batchTestLabels = self.orig_labels_test[i*loadBT:(i+1)*loadBT, :]

			accuracy_test, orig_feature, cae_feature, accuracy_test_CAE  = self.sess.run([self.accuracy, self.out_feature, self.cae_out, self.accuracy_CAE],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
			allaccuracy += accuracy_test
			allaccuracyCAE += accuracy_test_CAE
			temp = np.sqrt(np.sum((orig_feature - cae_feature)**2))
			tempden = np.sqrt(np.sum(orig_feature**2))
			print(temp, tempden)
			cae_err += temp / tempden

		allaccuracy = allaccuracy / numE
		allaccuracyCAE = allaccuracyCAE / numE
		cae_err = cae_err / numE
		print("TESTING ACCURACY: %.8f" % (allaccuracy))
		print("TESTING ACCURACY CAE: %.8f" % (allaccuracyCAE))
		print("CAE Error: %.8f" % (cae_err))

	def validation_model(self):
		'''
		Implements the testing routine.
		Also computes the CAE loss for reconstruction accuracy

		'''

		self.saver = tf.train.Saver()
		self.load_separate(self.checkpoint_dir, self.model_dir, self.model_name, self.bottleneck, self.reduced_data)
		loadBT = 10
		numE = int(self.orig_data_val.shape[0]/loadBT)
		allaccuracy = 0
		cae_err = 0
		for i in range(numE):
			batchTestData = self.orig_data_val[i*loadBT:(i+1)*loadBT, :]
			batchTestLabels = self.orig_labels_val[i*loadBT:(i+1)*loadBT, :]

			accuracy_test, orig_feature, cae_feature = self.sess.run([self.accuracy, self.out_feature, self.cae_out],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
			allaccuracy += accuracy_test
			temp = np.sqrt(np.sum((orig_feature - cae_feature)**2))
			tempden = np.sqrt(np.sum(orig_feature**2))
			cae_err += temp / tempden

		allaccuracy = allaccuracy / numE
		cae_err = cae_err / numE
		print("VALIDATION ACCURACY: %.8f" % (allaccuracy))
		print("CAE Error: %.8f" % (cae_err))

	@property
	def model_dir(self):
		return "{}".format(
			self.model_name)


	def save(self, checkpoint_dir, model_dir, model_name, step):
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir, model_name)
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name + '.model'), global_step=step)

	def save_separate(self, checkpoint_dir, model_dir, model_name, bt, rd, step):
		model_name = model_name + '_' + str(bt) + '_' + str(rd)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir, model_name)
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name + '.model'), global_step=step)

	def load(self, checkpoint_dir, model_dir, model_name):
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir, model_name)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		print('Loading Checkpoint ...')
		print(ckpt_name)
		self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

	def load_separate(self, checkpoint_dir, model_dir, model_name, bt, rd):
		model_name = model_name + '_' + str(bt) + '_' + str(rd)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir, model_name)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		print('Loading Checkpoint ...')
		print(ckpt_name)
		self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

class MNIST_CAEL1Norm_Classification:

	'''
	This class defines the state of the art MNIST classifier.
	With the case one variation, hard constraint bottleneck
	It's network architecture
	It's graph building function
	It's training and testing
	'''

	def __init__(self, sess, xmlfilename):

		''' Initialization of variables '''
		self.sess = sess

		tree=ET.parse(xmlfilename)
		root = tree.getroot()
		# self.separator_position = int(readSpecificTagXML(root, 'separator_position'))
		self.checkpoint_dir = ut.readSpecificTagXML(root, 'checkpoint_path')
		self.result_dir = ut.readSpecificTagXML(root, 'result_dir')
		self.log_dir = ut.readSpecificTagXML(root, 'log_dir')
		self.model_name = ut.readSpecificTagXML(root, 'model_name')
		self.learning_rate = float(ut.readSpecificTagXML(root, 'learning_rate'))
		self.l1_wt = float(ut.readSpecificTagXML(root, 'l1_wt'))
		self.batch_size = int(ut.readSpecificTagXML(root, 'batch_size'))
		self.epochs = int(ut.readSpecificTagXML(root, 'num_epochs'))
		self.print_iter = int(ut.readSpecificTagXML(root, 'print_iter'))
		self.save_iter = int(ut.readSpecificTagXML(root, 'save_iter'))
		# self.data_path = readSpecificTagXML(root, 'data_path')
		self.reduced_data = int(ut.readSpecificTagXML(root, 'reduced_data')) # number of data per label to use for training
		self.bottleneck = int(ut.readSpecificTagXML(root, 'bottleneck'))
		print('BOTTLENECK = ', self.bottleneck)
		self.cae_weight = float(ut.readSpecificTagXML(root, 'cae_weight'))
		self.burn_in_epochs = int(ut.readSpecificTagXML(root, 'burn_in'))
		# read and process the data sequence
		mnist = input_data.read_data_sets("../data/MNIST/", one_hot=True)
		[self.orig_data, self.orig_labels] = ut.reduceMNIST(mnist, self.reduced_data)
		print(self.orig_data.shape)
		self.orig_data_val = mnist.validation.images
		self.orig_labels_val = mnist.validation.labels
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		self.logFN = self.log_dir + '/logFile.csv'
		self.orig_data_test = mnist.test.images
		self.orig_labels_test = mnist.test.labels

		self.num_batches = int(self.orig_data.shape[0] / self.batch_size)

	def feature_extractor(self, x, phase_train, is_training=True, reuse=False):

		''' 
		The initial network which represents the feature extraction
		part of deep network

		'''
		with tf.variable_scope("feature_extractor", reuse=reuse):
			x_in = tf.reshape(x, [-1, 28, 28, 1])
			conv1_1 = ut.conv_layer(x_in, 1, 64, 'conv1_1', 3, phase_train)
			# conv1_2 = ut.conv_layer(conv1_1, 64, 128, 'conv1_2', 3, phase_train)
			with tf.name_scope('pool1'):
				h_pool1 = ut.max_pool_2x2_2d(conv1_1)
			# conv2_1 = ut.conv_layer(h_pool1, 128, 256, 'conv2_1', 3, phase_train)
			conv2_2 = ut.conv_layer(h_pool1, 64, 128, 'conv2_2', 3, phase_train)
			with tf.name_scope('pool2'):
				h_pool2 = ut.max_pool_2x2_2d(conv2_2)
				h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*128])
			
			return h_pool2_flat


	def classifier(self, x, phase_train, is_training=True, reuse=False):

		'''
		The end part of the deep network which takes in the extracted
		features and classifies them into digits

		'''
		with tf.variable_scope('classifier', reuse=reuse):
			with tf.name_scope('fc1'):
				W_fc1 = ut.weight_variable([7*7*128, 1024], 'fc1_wt')
				b_fc1 = ut.bias_variable([1024], 'fc1_bias')	
				h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

			with tf.name_scope('fc4'):
				W_fc4 = ut.weight_variable([1024, 10], 'fc4_wt')
				b_fc4 = ut.bias_variable([10], 'fc4_bias')
				out = tf.matmul(h_fc1, W_fc4) + b_fc4
			
			return out

	def CAE(self, x, phase_train, is_training=True, reuse=False):

		'''
		The external autoencoder based regualrization
		This is the Cooperative Autoencoder (CAE)
		'''

		with tf.variable_scope('CAE', reuse=reuse):
			with tf.name_scope('fc1-cae'):
				W_fc1_cae = ut.weight_variable([7*7*128, 1024], 'fc1_wt_cae')
				b_fc1_cae = ut.bias_variable([1024], 'fc1_bias_cae')
				h_fc1_cae = ut.parametric_relu(tf.matmul(x, W_fc1_cae) + b_fc1_cae)
			with tf.name_scope('fc1b-cae'):
				W_fc1b_cae = ut.weight_variable([1024, 512], 'fc1b_wt_cae')
				b_fc1b_cae = ut.bias_variable([512], 'fc1b_bias_cae')
				h_fc1b_cae = ut.parametric_relu(tf.matmul(h_fc1_cae, W_fc1b_cae) + b_fc1b_cae)
			with tf.name_scope('fc2-cae'):
				W_fc2_cae = ut.weight_variable([512, self.bottleneck], 'fc2_wt_cae')
				b_fc2_cae = ut.bias_variable([self.bottleneck], 'fc2_bias_cae')
				h_fc2_cae = ut.parametric_relu(tf.matmul(h_fc1b_cae, W_fc2_cae) + b_fc2_cae)
			with tf.name_scope('fc3-cae'):
				W_fc3_cae = ut.weight_variable([self.bottleneck, 512], 'fc3_wt_cae')
				b_fc3_cae = ut.bias_variable([512], 'fc3_bias_cae')
				h_fc3_cae = ut.parametric_relu(tf.matmul(h_fc2_cae, W_fc3_cae) + b_fc3_cae)
			with tf.name_scope('fc3b-cae'):
				W_fc3b_cae = ut.weight_variable([512, 1024], 'fc3b_wt_cae')
				b_fc3b_cae = ut.bias_variable([1024], 'fc3b_bias_cae')
				h_fc3b_cae = ut.parametric_relu(tf.matmul(h_fc3_cae, W_fc3b_cae) + b_fc3b_cae)
			with tf.name_scope('fc4-cae'):
				W_fc4_cae = ut.weight_variable([1024, 7*7*128], 'fc4_wt_cae')
				b_fc4_cae = ut.bias_variable([7*7*128], 'fc4_bias_cae')
				h_fc4_cae = ut.parametric_relu(tf.matmul(h_fc3b_cae, W_fc4_cae) + b_fc4_cae)

			return [h_fc4_cae, h_fc2_cae]

	def build_model(self):

		'''
		Builds the graph of the network, adds in the loss functions,
		defines placeholders.

		'''
		self.input_data = tf.placeholder(tf.float32, [None, 28*28], name='orig_data')
		self.labels = tf.placeholder(tf.float32, [None, 10], name='orig_labels')
		self.phase_train = tf.placeholder(tf.bool, name='phase')
		self.emptyPH = tf.placeholder(tf.float32, [None, self.bottleneck], name='emptyPH')
		self.out_feature = self.feature_extractor(self.input_data, self.phase_train, is_training=True, reuse=False)
		self.out_labels = self.classifier(self.out_feature, self.phase_train, is_training=True, reuse=False)
		[self.cae_out, self.bottleneck_value] = self.CAE(self.out_feature, self.phase_train, is_training=True, reuse=False)
		self.out_labels_fromCAE = self.classifier(self.cae_out, False, is_training=False, reuse=True)
		self.cae_loss = tf.reduce_mean(tf.nn.l2_loss(self.cae_out - self.out_feature)/ tf.nn.l2_loss(self.out_feature))
		self.featurenorm = tf.reduce_mean(tf.nn.l2_loss(self.out_feature))
		self.l1Loss = tf.reduce_mean(tf.losses.absolute_difference(self.bottleneck_value, self.emptyPH))
		# define the cross entropy loss
		# self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(self.out_labels), reduction_indices=[1]))
		self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_labels, labels=self.labels))
		correct_pred = tf.equal(tf.argmax(self.labels, 1), tf.argmax(tf.nn.softmax(self.out_labels), 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
		correct_pred_CAE = tf.equal(tf.argmax(self.labels, 1), tf.argmax(tf.nn.softmax(self.out_labels_fromCAE), 1))
		self.accuracy_CAE = tf.reduce_mean(tf.cast(correct_pred_CAE, tf.float32), name='accuracy-CAE')

		self.loss = self.cross_entropy + self.cae_weight*self.cae_loss + self.l1_wt*self.l1Loss

		'''Training function'''
		t_vars = tf.trainable_variables()
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optim_burnin = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cross_entropy, var_list=t_vars)
			self.optim = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss, var_list=t_vars)
			# self.optimall = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=t_vars)

	def train_model(self):

		''' 
		Implements the training routineself.

		'''
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		all_train_loss = []
		all_train_acc = []
		all_val_loss = []
		cae_val_loss = []
		cae_train_loss = []
		match_val_loss = []
		match_train_loss = []
		all_val_acc = []
		start_epoch = 0
		start_batch_id = 0
		counter = 1
		start_time = time.time()
		with open(self.logFN, 'w') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',')
			for epoch in range(start_epoch, self.epochs):
				[shufData, shufLables] = ut.training_shuffle_data(self.orig_data, self.orig_labels)
				for idx in range(start_batch_id, self.num_batches):
					[batch_data_train, batch_labels_train] = ut.get_batch_data(shufData, shufLables, idx, self.batch_size)
					[batch_data_val, batch_labels_val] = ut.get_batch_data(self.orig_data_val, self.orig_labels_val, idx, self.batch_size)
					batch_emptyPH_tr = np.zeros([batch_labels_train.shape[0], self.bottleneck])
					batch_emptyPH_val = np.zeros([batch_labels_val.shape[0], self.bottleneck])
					if epoch > self.burn_in_epochs:
						self.optim.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True, self.emptyPH:batch_emptyPH_tr} )
					else:
						self.optim_burnin.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True, self.emptyPH:batch_emptyPH_tr} )

					counter += 1
					# training accuracy

					if np.mod(counter, self.print_iter) == 0:
						# validation accuracy
						accuracy_train, match_train, cae_train, featureNT, tot_loss = self.sess.run([self.accuracy, self.cross_entropy, self.cae_loss, self.l1Loss, self.loss],
							feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:False, self.emptyPH:batch_emptyPH_tr})

						accuracy_val, match_val, cae_val, featureNV, tot_val = self.sess.run([self.accuracy, self.cross_entropy, self.cae_loss, self.l1Loss, self.loss],
							feed_dict={self.input_data: batch_data_val, self.labels: batch_labels_val, self.phase_train:False, self.emptyPH:batch_emptyPH_val})	
						l = [epoch, idx, accuracy_train, accuracy_val, match_train, match_val, cae_train, cae_val, featureNT, featureNV]
						spamwriter.writerow(l)
						print("TRAINING : Epoch: [%2d] [%4d/%4d] time: %4.4f, accuracy: %.8f, Match-Loss: %.8f, CAE-Loss: %.8f, l1-norm: %.8f" %(epoch, idx, self.num_batches, time.time() - start_time, accuracy_train, match_train, cae_train, featureNT))
						print("VALIDATION : Epoch: [%2d] [%4d/%4d] time: %4.4f, accuracy: %.8f, Match-Loss: %.8f, CAE-Loss: %.8f, l1-norm: %.8f" %(epoch, idx, self.num_batches, time.time() - start_time, accuracy_val, match_val, cae_val, featureNV))

				if np.mod(epoch, self.save_iter) == 0:
					# save the check point
					self.save_separate(self.checkpoint_dir, self.model_dir, self.model_name, self.bottleneck, self.reduced_data, counter)


	def test_model(self):

		'''
		Implements the testing routine.
		Also computes the CAE loss for reconstruction accuracy

		'''

		self.saver = tf.train.Saver()
		self.load_separate(self.checkpoint_dir, self.model_dir, self.model_name, self.bottleneck, self.reduced_data)
		loadBT = 10
		numE = int(self.orig_data_test.shape[0]/loadBT)
		allaccuracy = 0
		allaccuracyCAE = 0
		cae_err = 0
		for i in range(numE):
			batchTestData = self.orig_data_test[i*loadBT:(i+1)*loadBT, :]
			batchTestLabels = self.orig_labels_test[i*loadBT:(i+1)*loadBT, :]
			batch_emptyPH_test = np.zeros([batchTestLabels.shape[0], self.bottleneck])
			accuracy_test, orig_feature, cae_feature, accuracy_test_CAE  = self.sess.run([self.accuracy, self.out_feature, self.cae_out, self.accuracy_CAE],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False, self.emptyPH:batch_emptyPH_test})
			allaccuracy += accuracy_test
			allaccuracyCAE += accuracy_test_CAE
			temp = np.sqrt(np.sum((orig_feature - cae_feature)**2))
			tempden = np.sqrt(np.sum(orig_feature**2))
			print(temp, tempden)
			cae_err += temp / tempden

		allaccuracy = allaccuracy / numE
		allaccuracyCAE = allaccuracyCAE / numE
		cae_err = cae_err / numE
		print("TESTING ACCURACY: %.8f" % (allaccuracy))
		print("TESTING ACCURACY CAE: %.8f" % (allaccuracyCAE))
		print("CAE Error: %.8f" % (cae_err))
		return allaccuracy

	def validation_model(self):
		'''
		Implements the testing routine.
		Also computes the CAE loss for reconstruction accuracy

		'''

		self.saver = tf.train.Saver()
		self.load_separate(self.checkpoint_dir, self.model_dir, self.model_name, self.bottleneck, self.reduced_data)
		loadBT = 10
		numE = int(self.orig_data_val.shape[0]/loadBT)
		allaccuracy = 0
		cae_err = 0
		for i in range(numE):
			batchTestData = self.orig_data_val[i*loadBT:(i+1)*loadBT, :]
			batchTestLabels = self.orig_labels_val[i*loadBT:(i+1)*loadBT, :]

			accuracy_test, orig_feature, cae_feature = self.sess.run([self.accuracy, self.out_feature, self.cae_out],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
			allaccuracy += accuracy_test
			temp = np.sqrt(np.sum((orig_feature - cae_feature)**2))
			tempden = np.sqrt(np.sum(orig_feature**2))
			cae_err += temp / tempden

		allaccuracy = allaccuracy / numE
		cae_err = cae_err / numE
		print("VALIDATION ACCURACY: %.8f" % (allaccuracy))
		print("CAE Error: %.8f" % (cae_err))

	@property
	def model_dir(self):
		return "{}".format(
			self.model_name)


	def save(self, checkpoint_dir, model_dir, model_name, step):
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir, model_name)
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name + '.model'), global_step=step)

	def save_separate(self, checkpoint_dir, model_dir, model_name, bt, rd, step):
		model_name = model_name + '_' + str(bt) + '_' + str(rd)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir, model_name)
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name + '.model'), global_step=step)

	def load(self, checkpoint_dir, model_dir, model_name):
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir, model_name)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		print('Loading Checkpoint ...')
		print(ckpt_name)
		self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

	def load_separate(self, checkpoint_dir, model_dir, model_name, bt, rd):
		model_name = model_name + '_' + str(bt) + '_' + str(rd)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir, model_name)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		print('Loading Checkpoint ...')
		print(ckpt_name)
		self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))