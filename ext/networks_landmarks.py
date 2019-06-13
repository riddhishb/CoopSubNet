##########################################################################
#																		 #
# Desc : All network architectures for cooperative networks 			 #
# Author : Riddhish Bhalodia											 #
# Institution : Scientific Computing and Imaging Institute				 #
# Date : 20th January 2019												 #
#																		 #
##########################################################################

import numpy as np
import tensorflow as tf
import utils as ut
import csv
import os
import xml.etree.ElementTree as ET
# from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt
import tarfile
from six.moves import urllib


class Landmark_Base_Classification:

	'''
	This class defines the state of the art CIFAR 10 classifier.
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
		self.num_landmarks = int(ut.readSpecificTagXML(root, 'num_landmarks'))
		self.imgSize = np.array(ut.readSpecificTagXML(root, 'input_dims').split(" "), dtype=np.int64)
		# self.data_path = readSpecificTagXML(root, 'data_path')
		self.landmarkImages = ut.readSpecificTagXML(root, 'landmark_images')
		self.landmarkPoints = ut.readSpecificTagXML(root, 'landmark_points')
		# read and process the data sequence
		data_all = ut.landmarkDataPartition(self.landmarkImages, self.landmarkPoints, self.imgSize, self.num_landmarks)
		self.trainImages = data_all.trainImages(1)
		self.trainPoints = data_all.trainPoints(1)
		self.valImages = data_all.valImages(1)
		self.valPoints = data_all.valPoints(1)
		self.testImages = data_all.testImages(1)
		self.testPoints = data_all.testPoints(1)
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		self.logFN = self.log_dir + '/logFile.csv'
		self.num_batches = int(self.trainImages.shape[0] / self.batch_size)

	def feature_extractor(self, x, phase_train, is_training=True, reuse=False):

		''' 
		The initial network which represents the feature extraction
		part of deep network

		'''
		with tf.variable_scope("feature_extractor", reuse=reuse):
			x_in = tf.reshape(x, [-1, self.imgSize[0], self.imgSize[1], 3])

			conv1_1 = ut.conv_layer(x_in, 3, 8, 'conv1_1', 3, phase_train)
			# conv1_2 = ut.conv_layer(conv1_1, 64, 128, 'conv1_2', 3, phase_train)
			with tf.name_scope('pool1'):
				h_pool1 = ut.max_pool_2x2_2d(conv1_1)
			# conv2_1 = ut.conv_layer(h_pool1, 128, 256, 'conv2_1', 3, phase_train)
			conv2_1 = ut.conv_layer(h_pool1, 8, 16, 'conv2_1', 3, phase_train)
			with tf.name_scope('pool2'):
				h_pool2 = ut.max_pool_2x2_2d(conv2_1)
			conv3_1 = ut.conv_layer(h_pool2, 16, 32, 'conv3_1', 5, phase_train)
			with tf.name_scope('pool3'):
				h_pool3 = ut.max_pool_2x2_2d(conv3_1)
			conv4_1 = ut.conv_layer(h_pool3, 32, 64, 'conv4_1', 5, phase_train)
			with tf.name_scope('pool4'):
				h_pool4 = ut.max_pool_2x2_2d(conv4_1)
			conv5_1 = ut.conv_layer(h_pool4, 64, 128, 'conv5_1', 5, phase_train)
			with tf.name_scope('pool5'):
				h_pool5 = ut.max_pool_2x2_2d(conv5_1)
				h_pool5_flat = tf.reshape(h_pool5, [-1, int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0))])
				print(h_pool5.shape, h_pool5_flat.shape)
			return h_pool5_flat

	def classifier(self, x, phase_train, is_training=True, reuse=False):

		'''
		The end part of the deep network which takes in the extracted
		features and classifies them into digits

		'''
		with tf.variable_scope('classifier', reuse=reuse):
			with tf.name_scope('fc1'):
				W_fc1 = ut.weight_variable([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), 256], 'fc1_wt')
				b_fc1 = ut.bias_variable([256], 'fc1_bias')
				h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
			# with tf.name_scope('fc2'):
			# 	W_fc2 = ut.weight_variable([128, 256], 'fc2_wt')
			# 	b_fc2 = ut.bias_variable([256], 'fc2_bias')
			# 	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
			# with tf.name_scope('fc3'):
			# 	W_fc3 = ut.weight_variable([256, 512], 'fc3_wt')
			# 	b_fc3 = ut.bias_variable([512], 'fc3_bias')
			# 	h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
			# with tf.name_scope('fc4'):
			# 	W_fc4 = ut.weight_variable([512, 1024], 'fc4_wt')
			# 	b_fc4 = ut.bias_variable([1024], 'fc4_bias')
			# 	h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)
			with tf.name_scope('fc5'):
				W_fc5 = ut.weight_variable([256, self.num_landmarks*2], 'fc5_wt')
				b_fc5 = ut.bias_variable([self.num_landmarks*2], 'fc5_bias')
				out = tf.matmul(h_fc1, W_fc5) + b_fc5
			return out

	def build_model(self):

		'''
		Builds the graph of the network, adds in the loss functions,
		defines placeholders.

		'''
		self.input_data = tf.placeholder(tf.float32, [None, self.imgSize[0]*self.imgSize[1]*3], name='orig_data')
		self.labels = tf.placeholder(tf.float32, [None, self.num_landmarks*2], name='orig_labels')
		self.phase_train = tf.placeholder(tf.bool, name='phase')

		self.out_feature = self.feature_extractor(self.input_data, self.phase_train, is_training=True, reuse=False)
		self.out_labels = self.classifier(self.out_feature, self.phase_train, is_training=True, reuse=False)
		
		# define the cross entropy loss
		# self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_labels, labels=self.labels))
		# correct_pred = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.out_labels, 1))
		# self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
		self.squaredLoss = tf.reduce_mean((self.out_labels - self.labels)**2)

		'''Training function'''
		t_vars = tf.trainable_variables()
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.squaredLoss, var_list=t_vars)

	def train_model(self):

		''' 
		Implements the training routineself.

		'''
		print("Here")
		self.saver = tf.train.Saver()
		print("Here")
		self.sess.run(tf.global_variables_initializer())
		print("Here")
		all_train_loss = []
		all_train_acc = []
		all_val_loss = []
		all_val_acc = []
		start_epoch = 0
		start_batch_id = 0
		counter = 1
		start_time = time.time()
		
		with open(self.logFN, 'w') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',')
			for epoch in range(start_epoch, self.epochs):
				print("Here")
				[shufData, shufLables] = ut.training_shuffle_data(self.trainImages, self.trainPoints)
				total_accuracy_train = 0
				total_loss_train = 0
				total_accuracy_val = 0
				total_loss_val = 0

				for idx in range(start_batch_id, self.num_batches):
					# print(idx)
					[batch_data_train, batch_labels_train] = ut.get_batch_data(shufData, shufLables, idx, self.batch_size)
					self.optim.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True} )

					counter += 1
					accuracy_train, out_train = self.sess.run([self.squaredLoss, self.out_labels],
							feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:False})

					# total_loss_val += match_val
					total_accuracy_train += accuracy_train
					# total_accuracy_val += accuracy_val

				# total_loss_val = total_loss_val / self.num_batches
				total_accuracy_train = total_accuracy_train / self.num_batches
				print("TRAINING : Epoch: [%2d] time: %4.4f, error: %.8f" %(epoch, time.time() - start_time, total_accuracy_train))
				# total_accuracy_val = total_accuracy_val / self.num_batches
				# [total_accuracy_val, total_loss_val, out_val] = self.sess.run([self.accuracy, self.cross_entropy, self.out_labels],
				# 			feed_dict={self.input_data: self.orig_data_val, self.labels: self.orig_labels_val, self.phase_train:False})	
				# print(total_accuracy_val, total_loss_val)
				# print("VALIDATION : Epoch: [%2d] time: %4.4f, accuracy: %.8f, Match-Loss: %.8f," %(epoch, time.time() - start_time, total_accuracy_val, total_loss_val))
				all_train_acc.append(total_accuracy_train)
				# all_val_acc.append(total_accuracy_val)
				# all_val_loss.append(total_loss_val)

				if np.mod(epoch, self.save_iter) == 0:
					# save the check point
					self.save(self.checkpoint_dir, self.model_dir, self.model_name, counter)

		X_features = np.zeros([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), self.trainImages.shape[0]])
		for i in range(self.trainImages.shape[0]):
			batchTestData = self.trainImages[i, :].reshape(1, 3*self.imgSize[0]*self.imgSize[1])
			batchTestLabels = self.trainPoints[i, :].reshape(1, self.num_landmarks*2)

			temp = self.sess.run([self.out_feature],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
			# print(np.array(temp).shape)
			temp = np.array(temp)
			X_features[:, i]= temp[0,0,:]

		np.save('../experiments/CIFARCase1/Features/features_train.npy', X_features)
		np.save('../experiments/CIFARCase1/all_train_accuracy.npy', all_train_acc)

	def test_model(self):

		'''
		Implements the testing routine.

		'''

		self.saver = tf.train.Saver()
		self.load(self.checkpoint_dir, self.model_dir, self.model_name)
		loadBT = 1
		numE = int(self.testImages.shape[0]/loadBT)
		allaccuracy = 0
		np.save(os.path.join(self.result_dir, self.model_name) + 'testImages.npy', self.testImages)
		np.save(os.path.join(self.result_dir, self.model_name) + 'testPoints.npy', self.testPoints)
		outTestPoints = np.zeros([self.testPoints.shape[0], self.testPoints.shape[1]])
		for i in range(numE):
			batchTestData = self.testImages[i*loadBT:(i+1)*loadBT, :]
			batchTestLabels = self.testPoints[i*loadBT:(i+1)*loadBT, :]

			accuracy_test, out_test = self.sess.run([self.squaredLoss, self.out_labels],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
			allaccuracy += accuracy_test
			outTestPoints[i, ...] = out_test
		np.save(os.path.join(self.result_dir, self.model_name) + 'outputPoints.npy', outTestPoints)
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


class Landmark_Case2_Classification:

	'''
	This class defines the state of the art CIFAR 10 classifier.
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
		self.bottleneck = int(ut.readSpecificTagXML(root, 'bottleneck'))
		self.num_landmarks = int(ut.readSpecificTagXML(root, 'num_landmarks'))
		self.imgSize = np.array(ut.readSpecificTagXML(root, 'input_dims').split(" "), dtype=np.int64)
		# self.data_path = readSpecificTagXML(root, 'data_path')
		self.landmarkImages = ut.readSpecificTagXML(root, 'landmark_images')
		self.landmarkPoints = ut.readSpecificTagXML(root, 'landmark_points')
		# read and process the data sequence
		data_all = ut.landmarkDataPartition(self.landmarkImages, self.landmarkPoints, self.imgSize, self.num_landmarks)
		self.trainImages = data_all.trainImages(1)
		self.trainPoints = data_all.trainPoints(1)
		self.valImages = data_all.valImages(1)
		self.valPoints = data_all.valPoints(1)
		self.testImages = data_all.testImages(1)
		self.testPoints = data_all.testPoints(1)
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		self.logFN = self.log_dir + '/logFile.csv'
		self.num_batches = int(self.trainImages.shape[0] / self.batch_size)

	def feature_extractor(self, x, phase_train, is_training=True, reuse=False):

		''' 
		The initial network which represents the feature extraction
		part of deep network

		'''
		with tf.variable_scope("feature_extractor", reuse=reuse):
			x_in = tf.reshape(x, [-1, self.imgSize[0], self.imgSize[1], 3])

			conv1_1 = ut.conv_layer(x_in, 3, 8, 'conv1_1', 3, phase_train)
			# conv1_2 = ut.conv_layer(conv1_1, 64, 128, 'conv1_2', 3, phase_train)
			with tf.name_scope('pool1'):
				h_pool1 = ut.max_pool_2x2_2d(conv1_1)
			# conv2_1 = ut.conv_layer(h_pool1, 128, 256, 'conv2_1', 3, phase_train)
			conv2_1 = ut.conv_layer(h_pool1, 8, 16, 'conv2_1', 3, phase_train)
			with tf.name_scope('pool2'):
				h_pool2 = ut.max_pool_2x2_2d(conv2_1)
			conv3_1 = ut.conv_layer(h_pool2, 16, 32, 'conv3_1', 5, phase_train)
			with tf.name_scope('pool3'):
				h_pool3 = ut.max_pool_2x2_2d(conv3_1)
			conv4_1 = ut.conv_layer(h_pool3, 32, 64, 'conv4_1', 5, phase_train)
			with tf.name_scope('pool4'):
				h_pool4 = ut.max_pool_2x2_2d(conv4_1)
			conv5_1 = ut.conv_layer(h_pool4, 64, 128, 'conv5_1', 5, phase_train)
			with tf.name_scope('pool5'):
				h_pool5 = ut.max_pool_2x2_2d(conv5_1)
				h_pool5_flat = tf.reshape(h_pool5, [-1, int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0))])
				print(h_pool5.shape, h_pool5_flat.shape)
			return h_pool5_flat

	def classifier(self, x, phase_train, is_training=True, reuse=False):

		'''
		The end part of the deep network which takes in the extracted
		features and classifies them into digits

		'''
		with tf.variable_scope('classifier', reuse=reuse):
			with tf.name_scope('fc1'):
				W_fc1 = ut.weight_variable([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), 256], 'fc1_wt')
				b_fc1 = ut.bias_variable([256], 'fc1_bias')
				h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
			with tf.name_scope('fc2'):
				W_fc2 = ut.weight_variable([256, self.bottleneck], 'fc2_wt')
				b_fc2 = ut.bias_variable([self.bottleneck], 'fc2_bias')
				h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
			with tf.name_scope('fc3'):
				W_fc3 = ut.weight_variable([self.bottleneck, 256], 'fc3_wt')
				b_fc3 = ut.bias_variable([256], 'fc3_bias')
				h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
			with tf.name_scope('fc5'):
				W_fc5 = ut.weight_variable([256, self.num_landmarks*2], 'fc5_wt')
				b_fc5 = ut.bias_variable([self.num_landmarks*2], 'fc5_bias')
				out = tf.matmul(h_fc3, W_fc5) + b_fc5
			return out

	def build_model(self):

		'''
		Builds the graph of the network, adds in the loss functions,
		defines placeholders.

		'''
		self.input_data = tf.placeholder(tf.float32, [None, self.imgSize[0]*self.imgSize[1]*3], name='orig_data')
		self.labels = tf.placeholder(tf.float32, [None, self.num_landmarks*2], name='orig_labels')
		self.phase_train = tf.placeholder(tf.bool, name='phase')

		self.out_feature = self.feature_extractor(self.input_data, self.phase_train, is_training=True, reuse=False)
		self.out_labels = self.classifier(self.out_feature, self.phase_train, is_training=True, reuse=False)
		
		# define the cross entropy loss
		# self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_labels, labels=self.labels))
		# correct_pred = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.out_labels, 1))
		# self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
		self.squaredLoss = tf.reduce_mean((self.out_labels - self.labels)**2)

		'''Training function'''
		t_vars = tf.trainable_variables()
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.squaredLoss, var_list=t_vars)

	def train_model(self):

		''' 
		Implements the training routineself.

		'''
		print("Here")
		self.saver = tf.train.Saver()
		print("Here")
		self.sess.run(tf.global_variables_initializer())
		print("Here")
		all_train_loss = []
		all_train_acc = []
		all_val_loss = []
		all_val_acc = []
		start_epoch = 0
		start_batch_id = 0
		counter = 1
		start_time = time.time()
		
		with open(self.logFN, 'w') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',')
			for epoch in range(start_epoch, self.epochs):
				print("Here")
				[shufData, shufLables] = ut.training_shuffle_data(self.trainImages, self.trainPoints)
				total_accuracy_train = 0
				total_loss_train = 0
				total_accuracy_val = 0
				total_loss_val = 0

				for idx in range(start_batch_id, self.num_batches):
					# print(idx)
					[batch_data_train, batch_labels_train] = ut.get_batch_data(shufData, shufLables, idx, self.batch_size)
					self.optim.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True} )

					counter += 1
					accuracy_train, out_train = self.sess.run([self.squaredLoss, self.out_labels],
							feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:False})

					# total_loss_val += match_val
					total_accuracy_train += accuracy_train
					# total_accuracy_val += accuracy_val

				# total_loss_val = total_loss_val / self.num_batches
				total_accuracy_train = total_accuracy_train / self.num_batches
				print("TRAINING : Epoch: [%2d] time: %4.4f, error: %.8f" %(epoch, time.time() - start_time, total_accuracy_train))
				# total_accuracy_val = total_accuracy_val / self.num_batches
				# [total_accuracy_val, total_loss_val, out_val] = self.sess.run([self.accuracy, self.cross_entropy, self.out_labels],
				# 			feed_dict={self.input_data: self.orig_data_val, self.labels: self.orig_labels_val, self.phase_train:False})	
				# print(total_accuracy_val, total_loss_val)
				# print("VALIDATION : Epoch: [%2d] time: %4.4f, accuracy: %.8f, Match-Loss: %.8f," %(epoch, time.time() - start_time, total_accuracy_val, total_loss_val))
				all_train_acc.append(total_accuracy_train)
				# all_val_acc.append(total_accuracy_val)
				# all_val_loss.append(total_loss_val)

				if np.mod(epoch, self.save_iter) == 0:
					# save the check point
					self.save(self.checkpoint_dir, self.model_dir, self.model_name, counter)

	def test_model(self):

		'''
		Implements the testing routine.

		'''

		self.saver = tf.train.Saver()
		self.load(self.checkpoint_dir, self.model_dir, self.model_name)
		loadBT = 1
		numE = int(self.testImages.shape[0]/loadBT)
		allaccuracy = 0
		np.save(os.path.join(self.result_dir, self.model_name) + 'testImages.npy', self.testImages)
		np.save(os.path.join(self.result_dir, self.model_name) + 'testPoints.npy', self.testPoints)
		outTestPoints = np.zeros([self.testPoints.shape[0], self.testPoints.shape[1]])
		for i in range(numE):
			batchTestData = self.testImages[i*loadBT:(i+1)*loadBT, :]
			batchTestLabels = self.testPoints[i*loadBT:(i+1)*loadBT, :]

			accuracy_test, out_test = self.sess.run([self.squaredLoss, self.out_labels],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
			allaccuracy += accuracy_test
			outTestPoints[i, ...] = out_test
		np.save(os.path.join(self.result_dir, self.model_name) + 'outputPoints.npy', outTestPoints)
		allaccuracy = allaccuracy / numE
		print("TESTING ACCURACY: %.8f" % (allaccuracy))
		return allaccuracy

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


class Landmark_l2_Classification:

	'''
	This class defines the state of the art CIFAR 10 classifier.
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
		self.reg_wt = float(ut.readSpecificTagXML(root, 'reg_param'))
		self.batch_size = int(ut.readSpecificTagXML(root, 'batch_size'))
		self.epochs = int(ut.readSpecificTagXML(root, 'num_epochs'))
		self.print_iter = int(ut.readSpecificTagXML(root, 'print_iter'))
		self.save_iter = int(ut.readSpecificTagXML(root, 'save_iter'))
		self.num_landmarks = int(ut.readSpecificTagXML(root, 'num_landmarks'))
		self.imgSize = np.array(ut.readSpecificTagXML(root, 'input_dims').split(" "), dtype=np.int64)
		# self.data_path = readSpecificTagXML(root, 'data_path')
		self.landmarkImages = ut.readSpecificTagXML(root, 'landmark_images')
		self.landmarkPoints = ut.readSpecificTagXML(root, 'landmark_points')
		# read and process the data sequence
		data_all = ut.landmarkDataPartition(self.landmarkImages, self.landmarkPoints, self.imgSize, self.num_landmarks)
		self.trainImages = data_all.trainImages(1)
		self.trainPoints = data_all.trainPoints(1)
		self.valImages = data_all.valImages(1)
		self.valPoints = data_all.valPoints(1)
		self.testImages = data_all.testImages(1)
		self.testPoints = data_all.testPoints(1)
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		self.logFN = self.log_dir + '/logFile.csv'
		self.num_batches = int(self.trainImages.shape[0] / self.batch_size)

	def feature_extractor(self, x, phase_train, is_training=True, reuse=False):

		''' 
		The initial network which represents the feature extraction
		part of deep network

		'''
		with tf.variable_scope("feature_extractor", reuse=reuse):
			x_in = tf.reshape(x, [-1, self.imgSize[0], self.imgSize[1], 3])

			[conv1_1, regVal] = ut.conv_layer_reg(x_in, 3, 8, 'conv1_1', 3, phase_train)
			# conv1_2 = ut.conv_layer(conv1_1, 64, 128, 'conv1_2', 3, phase_train)
			with tf.name_scope('pool1'):
				h_pool1 = ut.max_pool_2x2_2d(conv1_1)
			# conv2_1 = ut.conv_layer(h_pool1, 128, 256, 'conv2_1', 3, phase_train)
			[conv2_1, temp] = ut.conv_layer_reg(h_pool1, 8, 16, 'conv2_1', 3, phase_train)
			regVal = regVal + temp
			with tf.name_scope('pool2'):
				h_pool2 = ut.max_pool_2x2_2d(conv2_1)
			[conv3_1, temp] = ut.conv_layer_reg(h_pool2, 16, 32, 'conv3_1', 5, phase_train)
			regVal = regVal + temp
			with tf.name_scope('pool3'):
				h_pool3 = ut.max_pool_2x2_2d(conv3_1)
			[conv4_1, temp] = ut.conv_layer_reg(h_pool3, 32, 64, 'conv4_1', 5, phase_train)
			regVal = regVal + temp
			with tf.name_scope('pool4'):
				h_pool4 = ut.max_pool_2x2_2d(conv4_1)
			[conv5_1, temp] = ut.conv_layer_reg(h_pool4, 64, 128, 'conv5_1', 5, phase_train)
			regVal = regVal + temp
			with tf.name_scope('pool5'):
				h_pool5 = ut.max_pool_2x2_2d(conv5_1)
				h_pool5_flat = tf.reshape(h_pool5, [-1, int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0))])
				print(h_pool5.shape, h_pool5_flat.shape)
			return [h_pool5_flat, regVal]

	def classifier(self, x, phase_train, regVal,  is_training=True, reuse=False):

		'''
		The end part of the deep network which takes in the extracted
		features and classifies them into digits

		'''
		with tf.variable_scope('classifier', reuse=reuse):
			with tf.name_scope('fc1'):
				W_fc1 = ut.weight_variable([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), 256], 'fc1_wt')
				b_fc1 = ut.bias_variable([256], 'fc1_bias')
				h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
				regVal = regVal + tf.nn.l2_loss(W_fc1)
			# with tf.name_scope('fc2'):
			# 	W_fc2 = ut.weight_variable([128, 256], 'fc2_wt')
			# 	b_fc2 = ut.bias_variable([256], 'fc2_bias')
			# 	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
			# with tf.name_scope('fc3'):
			# 	W_fc3 = ut.weight_variable([256, 512], 'fc3_wt')
			# 	b_fc3 = ut.bias_variable([512], 'fc3_bias')
			# 	h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
			# with tf.name_scope('fc4'):
			# 	W_fc4 = ut.weight_variable([512, 1024], 'fc4_wt')
			# 	b_fc4 = ut.bias_variable([1024], 'fc4_bias')
			# 	h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)
			with tf.name_scope('fc5'):
				W_fc5 = ut.weight_variable([256, self.num_landmarks*2], 'fc5_wt')
				b_fc5 = ut.bias_variable([self.num_landmarks*2], 'fc5_bias')
				out = tf.matmul(h_fc1, W_fc5) + b_fc5
				regVal = regVal + tf.nn.l2_loss(W_fc5)
			return [out, regVal]

	def build_model(self):

		'''
		Builds the graph of the network, adds in the loss functions,
		defines placeholders.

		'''
		self.input_data = tf.placeholder(tf.float32, [None, self.imgSize[0]*self.imgSize[1]*3], name='orig_data')
		self.labels = tf.placeholder(tf.float32, [None, self.num_landmarks*2], name='orig_labels')
		self.phase_train = tf.placeholder(tf.bool, name='phase')

		[self.out_feature, self.regVal] = self.feature_extractor(self.input_data, self.phase_train, is_training=True, reuse=False)
		[self.out_labels, self.regVal] = self.classifier(self.out_feature, self.phase_train, self.regVal, is_training=True, reuse=False)
		
		# define the cross entropy loss
		# self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_labels, labels=self.labels))
		# correct_pred = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.out_labels, 1))
		# self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
		self.squaredLoss = tf.reduce_mean((self.out_labels - self.labels)**2)
		self.loss = self.squaredLoss + self.reg_wt*self.regVal
		'''Training function'''
		t_vars = tf.trainable_variables()
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=t_vars)

	def train_model(self):

		''' 
		Implements the training routineself.

		'''
		print("Here")
		self.saver = tf.train.Saver()
		print("Here")
		self.sess.run(tf.global_variables_initializer())
		print("Here")
		all_train_loss = []
		all_train_acc = []
		all_val_loss = []
		all_val_acc = []
		start_epoch = 0
		start_batch_id = 0
		counter = 1
		start_time = time.time()
		
		with open(self.logFN, 'w') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',')
			for epoch in range(start_epoch, self.epochs):
				print("Here")
				[shufData, shufLables] = ut.training_shuffle_data(self.trainImages, self.trainPoints)
				total_accuracy_train = 0
				total_loss_train = 0
				total_accuracy_val = 0
				total_loss_val = 0

				for idx in range(start_batch_id, self.num_batches):
					# print(idx)
					[batch_data_train, batch_labels_train] = ut.get_batch_data(shufData, shufLables, idx, self.batch_size)
					self.optim.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True} )

					counter += 1
					accuracy_train, out_train = self.sess.run([self.squaredLoss, self.out_labels],
							feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:False})

					# total_loss_val += match_val
					total_accuracy_train += accuracy_train
					# total_accuracy_val += accuracy_val

				# total_loss_val = total_loss_val / self.num_batches
				total_accuracy_train = total_accuracy_train / self.num_batches
				print("TRAINING : Epoch: [%2d] time: %4.4f, error: %.8f" %(epoch, time.time() - start_time, total_accuracy_train))
				# total_accuracy_val = total_accuracy_val / self.num_batches
				# [total_accuracy_val, total_loss_val, out_val] = self.sess.run([self.accuracy, self.cross_entropy, self.out_labels],
				# 			feed_dict={self.input_data: self.orig_data_val, self.labels: self.orig_labels_val, self.phase_train:False})	
				# print(total_accuracy_val, total_loss_val)
				# print("VALIDATION : Epoch: [%2d] time: %4.4f, accuracy: %.8f, Match-Loss: %.8f," %(epoch, time.time() - start_time, total_accuracy_val, total_loss_val))
				all_train_acc.append(total_accuracy_train)
				# all_val_acc.append(total_accuracy_val)
				# all_val_loss.append(total_loss_val)

				if np.mod(epoch, self.save_iter) == 0:
					# save the check point
					self.save(self.checkpoint_dir, self.model_dir, self.model_name, counter)

		# X_features = np.zeros([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), self.trainImages.shape[0]])
		# for i in range(self.trainImages.shape[0]):
		# 	batchTestData = self.trainImages[i, :].reshape(1, 3*self.imgSize[0]*self.imgSize[1])
		# 	batchTestLabels = self.trainPoints[i, :].reshape(1, self.num_landmarks*2)

		# 	temp = self.sess.run([self.out_feature],
		# 		feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
		# 	# print(np.array(temp).shape)
		# 	temp = np.array(temp)
		# 	X_features[:, i]= temp[0,0,:]

		# np.save('../experiments/CIFARCase1/Features/features_train.npy', X_features)
		# np.save('../experiments/CIFARCase1/all_train_accuracy.npy', all_train_acc)

	def test_model(self):

		'''
		Implements the testing routine.

		'''

		self.saver = tf.train.Saver()
		self.load(self.checkpoint_dir, self.model_dir, self.model_name)
		loadBT = 1
		numE = int(self.testImages.shape[0]/loadBT)
		allaccuracy = 0
		np.save(os.path.join(self.result_dir, self.model_name) + 'testImages.npy', self.testImages)
		np.save(os.path.join(self.result_dir, self.model_name) + 'testPoints.npy', self.testPoints)
		outTestPoints = np.zeros([self.testPoints.shape[0], self.testPoints.shape[1]])
		for i in range(numE):
			batchTestData = self.testImages[i*loadBT:(i+1)*loadBT, :]
			batchTestLabels = self.testPoints[i*loadBT:(i+1)*loadBT, :]

			accuracy_test, out_test = self.sess.run([self.squaredLoss, self.out_labels],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
			allaccuracy += accuracy_test
			outTestPoints[i, ...] = out_test
		np.save(os.path.join(self.result_dir, self.model_name) + 'outputPoints.npy', outTestPoints)
		allaccuracy = allaccuracy / numE
		print("TESTING ACCURACY: %.8f" % (allaccuracy))
		# X_features = np.zeros([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), self.testImages.shape[0]])
		# for i in range(self.testImages.shape[0]):
		# 	batchTestData = self.testImages[i, :].reshape(1, 3*self.imgSize[0]*self.imgSize[1])
		# 	batchTestLabels = self.testPoints[i, :].reshape(1, self.num_landmarks*2)

		# 	temp = self.sess.run([self.out_feature],
		# 		feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
		# 	# print(np.array(temp).shape)
		# 	temp = np.array(temp)
		# 	X_features[:, i]= temp[0,0,:]
		# np.save('../experiments/CIFARCase1/Features/features_test.npy', X_features)
		return allaccuracy

	def PCA_test_model(self):
		
		'''
		Computes the PCA of the network feature space

		'''

		self.saver = tf.train.Saver()
		self.load(self.checkpoint_dir, self.model_dir, self.model_name)
		numE = self.orig_data_test.shape[0]
		print(numE)   
		allaccuracy = 0
		X_features = np.zeros([2*2*512, numE])
		Y_features = np.zeros([2*2*512, numE])
		for i in range(numE):
			batchTestData = self.orig_data_test[i, :].reshape(1, 3*32*32)
			batchTestLabels = self.orig_labels_test[i, :].reshape(1, self.numL)

			temp = self.sess.run([self.out_feature],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
			# print(np.array(temp).shape)
			temp = np.array(temp)
			X_features[:, i]= temp[0,0,:]
		np.save('../experiments/CIFARCase1/Features/features_test.npy', X_features)
		meanF = np.mean(X_features, 1)
		for i in range(numE):
			Y_features[..., i] = X_features[..., i] - meanF

		trickCovMat = np.dot(Y_features.T,Y_features) * 1.0/np.sqrt(numE-1)
		[s,v] = np.linalg.eigh(trickCovMat)
		eigs=s[::-1]
		y = np.cumsum(eigs) / np.sum(eigs)
		plt.figure()
		plt.plot(y)
		plt.savefig('Screeplot-CIFAR-Base-50.png')


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

class Landmark_Dropout_Classification:

	'''
	This class defines the state of the art CIFAR 10 classifier.
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
		self.num_landmarks = int(ut.readSpecificTagXML(root, 'num_landmarks'))
		self.keep_prob = float(ut.readSpecificTagXML(root, 'keep_prob'))
		self.imgSize = np.array(ut.readSpecificTagXML(root, 'input_dims').split(" "), dtype=np.int64)
		# self.data_path = readSpecificTagXML(root, 'data_path')
		self.landmarkImages = ut.readSpecificTagXML(root, 'landmark_images')
		self.landmarkPoints = ut.readSpecificTagXML(root, 'landmark_points')
		# read and process the data sequence
		data_all = ut.landmarkDataPartition(self.landmarkImages, self.landmarkPoints, self.imgSize, self.num_landmarks)
		self.trainImages = data_all.trainImages(1)
		self.trainPoints = data_all.trainPoints(1)
		self.valImages = data_all.valImages(1)
		self.valPoints = data_all.valPoints(1)
		self.testImages = data_all.testImages(1)
		self.testPoints = data_all.testPoints(1)
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		self.logFN = self.log_dir + '/logFile.csv'
		self.num_batches = int(self.trainImages.shape[0] / self.batch_size)

	def feature_extractor(self, x, phase_train, is_training=True, reuse=False):

		''' 
		The initial network which represents the feature extraction
		part of deep network

		'''
		with tf.variable_scope("feature_extractor", reuse=reuse):
			x_in = tf.reshape(x, [-1, self.imgSize[0], self.imgSize[1], 3])

			conv1_1 = ut.conv_layer(x_in, 3, 8, 'conv1_1', 3, phase_train)
			# conv1_2 = ut.conv_layer(conv1_1, 64, 128, 'conv1_2', 3, phase_train)
			with tf.name_scope('pool1'):
				h_pool1 = ut.max_pool_2x2_2d(conv1_1)
			# conv2_1 = ut.conv_layer(h_pool1, 128, 256, 'conv2_1', 3, phase_train)
			conv2_1 = ut.conv_layer(h_pool1, 8, 16, 'conv2_1', 3, phase_train)
			with tf.name_scope('pool2'):
				h_pool2 = ut.max_pool_2x2_2d(conv2_1)
			conv3_1 = ut.conv_layer(h_pool2, 16, 32, 'conv3_1', 5, phase_train)
			with tf.name_scope('pool3'):
				h_pool3 = ut.max_pool_2x2_2d(conv3_1)
			conv4_1 = ut.conv_layer(h_pool3, 32, 64, 'conv4_1', 5, phase_train)
			with tf.name_scope('pool4'):
				h_pool4 = ut.max_pool_2x2_2d(conv4_1)
			conv5_1 = ut.conv_layer(h_pool4, 64, 128, 'conv5_1', 5, phase_train)
			with tf.name_scope('pool5'):
				h_pool5 = ut.max_pool_2x2_2d(conv5_1)
				h_pool5_flat = tf.reshape(h_pool5, [-1, int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0))])
				print(h_pool5.shape, h_pool5_flat.shape)
			return h_pool5_flat

	def classifier(self, x, phase_train, keep_prob, is_training=True, reuse=False):

		'''
		The end part of the deep network which takes in the extracted
		features and classifies them into digits

		'''
		with tf.variable_scope('classifier', reuse=reuse):
			with tf.name_scope('dropout'):
				h_fc1_drop = tf.nn.dropout(x, keep_prob)

			with tf.name_scope('fc1'):
				W_fc1 = ut.weight_variable([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), 256], 'fc1_wt')
				b_fc1 = ut.bias_variable([256], 'fc1_bias')
				h_fc1 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc1) + b_fc1)
			# with tf.name_scope('fc2'):
			# 	W_fc2 = ut.weight_variable([128, 256], 'fc2_wt')
			# 	b_fc2 = ut.bias_variable([256], 'fc2_bias')
			# 	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
			# with tf.name_scope('fc3'):
			# 	W_fc3 = ut.weight_variable([256, 512], 'fc3_wt')
			# 	b_fc3 = ut.bias_variable([512], 'fc3_bias')
			# 	h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
			# with tf.name_scope('fc4'):
			# 	W_fc4 = ut.weight_variable([512, 1024], 'fc4_wt')
			# 	b_fc4 = ut.bias_variable([1024], 'fc4_bias')
			# 	h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)
			with tf.name_scope('fc5'):
				W_fc5 = ut.weight_variable([256, self.num_landmarks*2], 'fc5_wt')
				b_fc5 = ut.bias_variable([self.num_landmarks*2], 'fc5_bias')
				out = tf.matmul(h_fc1, W_fc5) + b_fc5
			return out

	def build_model(self):

		'''
		Builds the graph of the network, adds in the loss functions,
		defines placeholders.

		'''
		self.input_data = tf.placeholder(tf.float32, [None, self.imgSize[0]*self.imgSize[1]*3], name='orig_data')
		self.labels = tf.placeholder(tf.float32, [None, self.num_landmarks*2], name='orig_labels')
		self.phase_train = tf.placeholder(tf.bool, name='phase')
		self.kp = tf.placeholder(tf.float32)
		self.out_feature = self.feature_extractor(self.input_data, self.phase_train, is_training=True, reuse=False)
		self.out_labels = self.classifier(self.out_feature, self.phase_train, self.kp, is_training=True, reuse=False)
		
		# define the cross entropy loss
		# self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_labels, labels=self.labels))
		# correct_pred = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.out_labels, 1))
		# self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
		self.squaredLoss = tf.reduce_mean((self.out_labels - self.labels)**2)

		'''Training function'''
		t_vars = tf.trainable_variables()
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.squaredLoss, var_list=t_vars)

	def train_model(self):

		''' 
		Implements the training routineself.

		'''
		print("Here")
		self.saver = tf.train.Saver()
		print("Here")
		self.sess.run(tf.global_variables_initializer())
		print("Here")
		all_train_loss = []
		all_train_acc = []
		all_val_loss = []
		all_val_acc = []
		start_epoch = 0
		start_batch_id = 0
		counter = 1
		start_time = time.time()
		
		with open(self.logFN, 'w') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',')
			for epoch in range(start_epoch, self.epochs):
				print("Here")
				[shufData, shufLables] = ut.training_shuffle_data(self.trainImages, self.trainPoints)
				total_accuracy_train = 0
				total_loss_train = 0
				total_accuracy_val = 0
				total_loss_val = 0

				for idx in range(start_batch_id, self.num_batches):
					# print(idx)
					[batch_data_train, batch_labels_train] = ut.get_batch_data(shufData, shufLables, idx, self.batch_size)
					self.optim.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True, self.kp:self.keep_prob} )

					counter += 1
					accuracy_train, out_train = self.sess.run([self.squaredLoss, self.out_labels],
							feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:False, self.kp:self.keep_prob})

					# total_loss_val += match_val
					total_accuracy_train += accuracy_train
					# total_accuracy_val += accuracy_val

				# total_loss_val = total_loss_val / self.num_batches
				total_accuracy_train = total_accuracy_train / self.num_batches
				print("TRAINING : Epoch: [%2d] time: %4.4f, error: %.8f" %(epoch, time.time() - start_time, total_accuracy_train))
				# total_accuracy_val = total_accuracy_val / self.num_batches
				# [total_accuracy_val, total_loss_val, out_val] = self.sess.run([self.accuracy, self.cross_entropy, self.out_labels],
				# 			feed_dict={self.input_data: self.orig_data_val, self.labels: self.orig_labels_val, self.phase_train:False})	
				# print(total_accuracy_val, total_loss_val)
				# print("VALIDATION : Epoch: [%2d] time: %4.4f, accuracy: %.8f, Match-Loss: %.8f," %(epoch, time.time() - start_time, total_accuracy_val, total_loss_val))
				all_train_acc.append(total_accuracy_train)
				# all_val_acc.append(total_accuracy_val)
				# all_val_loss.append(total_loss_val)

				if np.mod(epoch, self.save_iter) == 0:
					# save the check point
					self.save(self.checkpoint_dir, self.model_dir, self.model_name, counter)

		X_features = np.zeros([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), self.trainImages.shape[0]])
		for i in range(self.trainImages.shape[0]):
			batchTestData = self.trainImages[i, :].reshape(1, 3*self.imgSize[0]*self.imgSize[1])
			batchTestLabels = self.trainPoints[i, :].reshape(1, self.num_landmarks*2)

			temp = self.sess.run([self.out_feature],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False, self.kp:self.keep_prob})
			# print(np.array(temp).shape)
			temp = np.array(temp)
			X_features[:, i]= temp[0,0,:]

		np.save('../experiments/CIFARCase1/Features/features_train.npy', X_features)
		np.save('../experiments/CIFARCase1/all_train_accuracy.npy', all_train_acc)

	def test_model(self):

		'''
		Implements the testing routine.

		'''

		self.saver = tf.train.Saver()
		self.load(self.checkpoint_dir, self.model_dir, self.model_name)
		loadBT = 1
		numE = int(self.testImages.shape[0]/loadBT)
		allaccuracy = 0
		np.save(os.path.join(self.result_dir, self.model_name) + 'testImages.npy', self.testImages)
		np.save(os.path.join(self.result_dir, self.model_name) + 'testPoints.npy', self.testPoints)
		outTestPoints = np.zeros([self.testPoints.shape[0], self.testPoints.shape[1]])
		for i in range(numE):
			batchTestData = self.testImages[i*loadBT:(i+1)*loadBT, :]
			batchTestLabels = self.testPoints[i*loadBT:(i+1)*loadBT, :]

			accuracy_test, out_test = self.sess.run([self.squaredLoss, self.out_labels],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False, self.kp:self.keep_prob})
			allaccuracy += accuracy_test
			outTestPoints[i, ...] = out_test
		np.save(os.path.join(self.result_dir, self.model_name) + 'outputPoints.npy', outTestPoints)
		allaccuracy = allaccuracy / numE
		print("TESTING ACCURACY: %.8f" % (allaccuracy))
		X_features = np.zeros([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), self.testImages.shape[0]])
		for i in range(self.testImages.shape[0]):
			batchTestData = self.testImages[i, :].reshape(1, 3*self.imgSize[0]*self.imgSize[1])
			batchTestLabels = self.testPoints[i, :].reshape(1, self.num_landmarks*2)

			temp = self.sess.run([self.out_feature],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False, self.kp:self.keep_prob})
			# print(np.array(temp).shape)
			temp = np.array(temp)
			X_features[:, i]= temp[0,0,:]
		np.save('../experiments/CIFARCase1/Features/features_test.npy', X_features)

	def PCA_test_model(self):
		
		'''
		Computes the PCA of the network feature space

		'''

		self.saver = tf.train.Saver()
		self.load(self.checkpoint_dir, self.model_dir, self.model_name)
		numE = self.orig_data_test.shape[0]
		print(numE)   
		allaccuracy = 0
		X_features = np.zeros([2*2*512, numE])
		Y_features = np.zeros([2*2*512, numE])
		for i in range(numE):
			batchTestData = self.orig_data_test[i, :].reshape(1, 3*32*32)
			batchTestLabels = self.orig_labels_test[i, :].reshape(1, self.numL)

			temp = self.sess.run([self.out_feature],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
			# print(np.array(temp).shape)
			temp = np.array(temp)
			X_features[:, i]= temp[0,0,:]
		np.save('../experiments/CIFARCase1/Features/features_test.npy', X_features)
		meanF = np.mean(X_features, 1)
		for i in range(numE):
			Y_features[..., i] = X_features[..., i] - meanF

		trickCovMat = np.dot(Y_features.T,Y_features) * 1.0/np.sqrt(numE-1)
		[s,v] = np.linalg.eigh(trickCovMat)
		eigs=s[::-1]
		y = np.cumsum(eigs) / np.sum(eigs)
		plt.figure()
		plt.plot(y)
		plt.savefig('Screeplot-CIFAR-Base-50.png')


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


class Landmark_CAE_Classification:

	'''
	This class defines the state of the art CIFAR 10 classifier.
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
		self.num_landmarks = int(ut.readSpecificTagXML(root, 'num_landmarks'))
		self.imgSize = np.array(ut.readSpecificTagXML(root, 'input_dims').split(" "), dtype=np.int64)
		# self.data_path = readSpecificTagXML(root, 'data_path')
		self.landmarkImages = ut.readSpecificTagXML(root, 'landmark_images')
		self.landmarkPoints = ut.readSpecificTagXML(root, 'landmark_points')
		self.bottleneck = int(ut.readSpecificTagXML(root, 'bottleneck'))
		self.noise_var = float(ut.readSpecificTagXML(root, 'noise_var'))
		self.cae_weight_init = float(ut.readSpecificTagXML(root, 'cae_weight_init'))
		self.cae_weight_final = float(ut.readSpecificTagXML(root, 'cae_weight_final')) 
		self.burn_in_epochs = int(ut.readSpecificTagXML(root, 'burn_in'))
		# read and process the data sequence
		data_all = ut.landmarkDataPartition(self.landmarkImages, self.landmarkPoints, self.imgSize, self.num_landmarks)
		self.trainImages = data_all.trainImages(1)
		self.trainPoints = data_all.trainPoints(1)
		self.valImages = data_all.valImages(1)
		self.valPoints = data_all.valPoints(1)
		self.testImages = data_all.testImages(1)
		self.testPoints = data_all.testPoints(1)
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		self.logFN = self.log_dir + '/logFile.csv'
		self.num_batches = int(self.trainImages.shape[0] / self.batch_size)

	def feature_extractor(self, x, phase_train, is_training=True, reuse=False):

		''' 
		The initial network which represents the feature extraction
		part of deep network

		'''
		with tf.variable_scope("feature_extractor", reuse=reuse):
			x_in = tf.reshape(x, [-1, self.imgSize[0], self.imgSize[1], 3])

			conv1_1 = ut.conv_layer(x_in, 3, 8, 'conv1_1', 3, phase_train)
			# conv1_2 = ut.conv_layer(conv1_1, 64, 128, 'conv1_2', 3, phase_train)
			with tf.name_scope('pool1'):
				h_pool1 = ut.max_pool_2x2_2d(conv1_1)
			# conv2_1 = ut.conv_layer(h_pool1, 128, 256, 'conv2_1', 3, phase_train)
			conv2_1 = ut.conv_layer(h_pool1, 8, 16, 'conv2_1', 3, phase_train)
			with tf.name_scope('pool2'):
				h_pool2 = ut.max_pool_2x2_2d(conv2_1)
			conv3_1 = ut.conv_layer(h_pool2, 16, 32, 'conv3_1', 5, phase_train)
			with tf.name_scope('pool3'):
				h_pool3 = ut.max_pool_2x2_2d(conv3_1)
			conv4_1 = ut.conv_layer(h_pool3, 32, 64, 'conv4_1', 5, phase_train)
			with tf.name_scope('pool4'):
				h_pool4 = ut.max_pool_2x2_2d(conv4_1)
			conv5_1 = ut.conv_layer(h_pool4, 64, 128, 'conv5_1', 5, phase_train)
			with tf.name_scope('pool5'):
				h_pool5 = ut.max_pool_2x2_2d(conv5_1)
				h_pool5_flat = tf.reshape(h_pool5, [-1, int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0))])
			return h_pool5_flat

	def classifier(self, x, phase_train, is_training=True, reuse=False):

		'''
		The end part of the deep network which takes in the extracted
		features and classifies them into digits

		'''
		with tf.variable_scope('classifier', reuse=reuse):
			with tf.name_scope('fc1'):
				W_fc1 = ut.weight_variable([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), 256], 'fc1_wt')
				b_fc1 = ut.bias_variable([256], 'fc1_bias')
				h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
			# with tf.name_scope('fc2'):
			# 	W_fc2 = ut.weight_variable([128, 256], 'fc2_wt')
			# 	b_fc2 = ut.bias_variable([256], 'fc2_bias')
			# 	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
			# with tf.name_scope('fc3'):
			# 	W_fc3 = ut.weight_variable([256, 512], 'fc3_wt')
			# 	b_fc3 = ut.bias_variable([512], 'fc3_bias')
			# 	h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
			# with tf.name_scope('fc4'):
			# 	W_fc4 = ut.weight_variable([512, 1024], 'fc4_wt')
			# 	b_fc4 = ut.bias_variable([1024], 'fc4_bias')
			# 	h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)
			with tf.name_scope('fc5'):
				W_fc5 = ut.weight_variable([256, self.num_landmarks*2], 'fc5_wt')
				b_fc5 = ut.bias_variable([self.num_landmarks*2], 'fc5_bias')
				out = tf.matmul(h_fc1, W_fc5) + b_fc5
			return out

	def CAE(self, x, phase_train, is_training=True, reuse=False):
		with tf.variable_scope('CAE', reuse=reuse):
			with tf.name_scope('fc1-cae'):
				W_fc1_cae = ut.weight_variable([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), 256], 'fc1_wt_cae')
				b_fc1_cae = ut.bias_variable([256], 'fc1_bias_cae')
				h_fc1_cae = ut.parametric_relu(tf.matmul(x, W_fc1_cae) + b_fc1_cae)
			with tf.name_scope('fc1b-cae'):
				W_fc1b_cae = ut.weight_variable([256, 128], 'fc1b_wt_cae')
				b_fc1b_cae = ut.bias_variable([128], 'fc1b_bias_cae')
				h_fc1b_cae = ut.parametric_relu(tf.matmul(h_fc1_cae, W_fc1b_cae) + b_fc1b_cae)
			# with tf.name_scope('fc1c-cae'):
			# 	W_fc1c_cae = ut.weight_variable([512, 256], 'fc1c_wt_cae')
			# 	b_fc1c_cae = ut.bias_variable([256], 'fc1c_bias_cae')
			# 	h_fc1c_cae = ut.parametric_relu(tf.matmul(h_fc1b_cae, W_fc1c_cae) + b_fc1c_cae)
			with tf.name_scope('fc2-cae'):
				W_fc2_cae = ut.weight_variable([128, self.bottleneck], 'fc2_wt_cae')
				b_fc2_cae = ut.bias_variable([self.bottleneck], 'fc2_bias_cae')
				h_fc2_cae = ut.parametric_relu(tf.matmul(h_fc1b_cae, W_fc2_cae) + b_fc2_cae)
			with tf.name_scope('fc3-cae'):
				W_fc3_cae = ut.weight_variable([self.bottleneck, 128], 'fc3_wt_cae')
				b_fc3_cae = ut.bias_variable([128], 'fc3_bias_cae')
				h_fc3_cae = ut.parametric_relu(tf.matmul(h_fc2_cae, W_fc3_cae) + b_fc3_cae)
			# with tf.name_scope('fc3c-cae'):
			# 	W_fc3c_cae = ut.weight_variable([256, 512], 'fc3c_wt_cae')
			# 	b_fc3c_cae = ut.bias_variable([512], 'fc3c_bias_cae')
			# 	h_fc3c_cae = ut.parametric_relu(tf.matmul(h_fc3_cae, W_fc3c_cae) + b_fc3c_cae)
			with tf.name_scope('fc3b-cae'):
				W_fc3b_cae = ut.weight_variable([128, 256], 'fc3b_wt_cae')
				b_fc3b_cae = ut.bias_variable([256], 'fc3b_bias_cae')
				h_fc3b_cae = ut.parametric_relu(tf.matmul(h_fc3_cae, W_fc3b_cae) + b_fc3b_cae)
			with tf.name_scope('fc4-cae'):
				W_fc4_cae = ut.weight_variable([256, int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0))], 'fc4_wt_cae')
				b_fc4_cae = ut.bias_variable([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0))], 'fc4_bias_cae')
				h_fc4_cae = ut.parametric_relu(tf.matmul(h_fc3b_cae, W_fc4_cae) + b_fc4_cae)
			return  [h_fc4_cae, h_fc2_cae]

	def build_model(self):

		'''
		Builds the graph of the network, adds in the loss functions,
		defines placeholders.

		'''
		self.input_data = tf.placeholder(tf.float32, [None, self.imgSize[0]*self.imgSize[1]*3], name='orig_data')
		self.labels = tf.placeholder(tf.float32, [None, self.num_landmarks*2], name='orig_labels')
		self.phase_train = tf.placeholder(tf.bool, name='phase')
		self.cae_weight = tf.placeholder(tf.float32, name='wt')
		self.out_feature = self.feature_extractor(self.input_data, self.phase_train, is_training=True, reuse=False)
		self.out_labels = self.classifier(self.out_feature, self.phase_train, is_training=True, reuse=False)
		self.noisy_features = self.out_feature + self.noise_var * tf.random_normal(tf.shape(self.out_feature), 0, 1, dtype=tf.float32)
		[self.cae_out, self.bottleneck_value] = self.CAE(self.noisy_features, self.phase_train, is_training=True, reuse=False)
		self.cae_loss = tf.reduce_mean(tf.nn.l2_loss(self.cae_out - self.out_feature)/ tf.nn.l2_loss(self.out_feature))
		# define the cross entropy loss
		# self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_labels, labels=self.labels))
		# correct_pred = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.out_labels, 1))
		# self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
		self.squaredLoss = tf.reduce_mean((self.out_labels - self.labels)**2)
		self.loss = self.squaredLoss + self.cae_weight*self.cae_loss
		'''Training function'''
		t_vars = tf.trainable_variables()
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optim_burnin = tf.train.AdamOptimizer(self.learning_rate).minimize(self.squaredLoss, var_list=t_vars)
			self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=t_vars)

	def train_model(self):

		''' 
		Implements the training routineself.

		'''
		print("Here")
		self.saver = tf.train.Saver()
		print("Here")
		self.sess.run(tf.global_variables_initializer())
		print("Here")
		all_train_loss = []
		all_train_acc = []
		all_val_loss = []
		all_val_acc = []
		start_epoch = 0
		start_batch_id = 0
		counter = 1
		start_time = time.time()
		
		with open(self.logFN, 'w') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',')
			for epoch in range(start_epoch, self.epochs):
				[shufData, shufLables] = ut.training_shuffle_data(self.trainImages, self.trainPoints)
				total_accuracy_train = 0
				total_loss_train = 0
				total_accuracy_val = 0
				total_loss_val = 0

				caeWT = self.cae_weight_init + (epoch/self.epochs)*(self.cae_weight_final - self.cae_weight_init)
				for idx in range(start_batch_id, self.num_batches):
					[batch_data_train, batch_labels_train] = ut.get_batch_data(shufData, shufLables, idx, self.batch_size)
					# [batch_data_val, batch_labels_val] = ut.get_batch_data(self.valImages, self.valPoints, idx, self.batch_size)
					if epoch > self.burn_in_epochs:
						self.optim.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True, self.cae_weight : caeWT} )
					else:
						self.optim_burnin.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True, self.cae_weight : caeWT} )

					counter += 1
					# training accuracy

					if np.mod(counter, self.print_iter) == 0:
						# validation accuracy
						accuracy_train, match_train, cae_train = self.sess.run([self.loss, self.squaredLoss, self.cae_loss],
							feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:False, self.cae_weight : caeWT})

						# accuracy_val, match_val, cae_val, featureNV, tot_val = self.sess.run([self.accuracy, self.cross_entropy, self.cae_loss, self.featurenorm, self.loss],
						# 	feed_dict={self.input_data: batch_data_val, self.labels: batch_labels_val, self.phase_train:False, self.cae_weight : caeWT})	
						# l = [epoch, idx, accuracy_train, accuracy_val, match_train, match_val, cae_train, cae_val, featureNT, featureNV]
						# spamwriter.writerow(l)
						print("TRAINING : Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, Match-Loss: %.8f, CAE-Loss: %.8f" %(epoch, idx, self.num_batches, time.time() - start_time, accuracy_train, match_train, cae_train))
						# print("VALIDATION : Epoch: [%2d] [%4d/%4d] time: %4.4f, accuracy: %.8f, Match-Loss: %.8f, CAE-Loss: %.8f, feature-norm: %.8f" %(epoch, idx, self.num_batches, time.time() - start_time, accuracy_val, match_val, cae_val, featureNV))
				# total_loss_val = total_loss_val / self.num_batches
				# total_accuracy_train = total_accuracy_train / self.num_batches
				# print("TRAINING : Epoch: [%2d] time: %4.4f, error: %.8f" %(epoch, time.time() - start_time, total_accuracy_train))
				# total_accuracy_val = total_accuracy_val / self.num_batches
				# [total_accuracy_val, total_loss_val, out_val] = self.sess.run([self.accuracy, self.cross_entropy, self.out_labels],
				# 			feed_dict={self.input_data: self.orig_data_val, self.labels: self.orig_labels_val, self.phase_train:False})	
				# print(total_accuracy_val, total_loss_val)
				# print("VALIDATION : Epoch: [%2d] time: %4.4f, accuracy: %.8f, Match-Loss: %.8f," %(epoch, time.time() - start_time, total_accuracy_val, total_loss_val))
				# all_train_acc.append(total_accuracy_train)
				# all_val_acc.append(total_accuracy_val)
				# all_val_loss.append(total_loss_val)

				if np.mod(epoch, self.save_iter) == 0:
					# save the check point
					self.save(self.checkpoint_dir, self.model_dir, self.model_name, counter)

		# X_features = np.zeros([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), self.trainImages.shape[0]])
		# for i in range(self.trainImages.shape[0]):
		# 	batchTestData = self.trainImages[i, :].reshape(1, 3*self.imgSize[0]*self.imgSize[1])
		# 	batchTestLabels = self.trainPoints[i, :].reshape(1, self.num_landmarks*2)

		# 	temp = self.sess.run([self.out_feature],
		# 		feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
		# 	# print(np.array(temp).shape)
		# 	temp = np.array(temp)
		# 	X_features[:, i]= temp[0,0,:]

		# np.save('../experiments/CIFARCase1/Features/features_train.npy', X_features)
		# np.save('../experiments/CIFARCase1/all_train_accuracy.npy', all_train_acc)

	def test_model(self):

		'''
		Implements the testing routine.

		'''

		self.saver = tf.train.Saver()
		self.load(self.checkpoint_dir, self.model_dir, self.model_name)
		loadBT = 1
		numE = int(self.testImages.shape[0]/loadBT)
		allaccuracy = 0
		allaccuracy_cae = 0
		allaccuracy_squared = 0
		np.save(os.path.join(self.result_dir, self.model_name) + 'testImages_CAE.npy', self.testImages)
		np.save(os.path.join(self.result_dir, self.model_name) + 'testPoints_CAE.npy', self.testPoints)
		outTestPoints = np.zeros([self.testPoints.shape[0], self.testPoints.shape[1]])
		for i in range(numE):
			batchTestData = self.testImages[i*loadBT:(i+1)*loadBT, :]
			batchTestLabels = self.testPoints[i*loadBT:(i+1)*loadBT, :]

			accuracy_test, squared_test, out_test, orig_feature, cae_feature = self.sess.run([self.loss, self.squaredLoss, self.out_labels, self.out_feature, self.cae_out],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False, self.cae_weight:self.cae_weight_init})
			allaccuracy += accuracy_test
			allaccuracy_squared += squared_test
			outTestPoints[i, ...] = out_test
			temp = np.sqrt(np.sum((orig_feature - cae_feature)**2))
			tempden = np.sqrt(np.sum(orig_feature**2))
			print(temp, tempden)
			allaccuracy_cae += temp / tempden
		np.save(os.path.join(self.result_dir, self.model_name) + 'outputPoints_CAE.npy', outTestPoints)
		allaccuracy = allaccuracy / numE
		print("TESTING ACCURACY: %.8f,  squaredLoss: %.8f, cae-LOss : %.8f" % (allaccuracy, allaccuracy_squared, allaccuracy_cae))
		# X_features = np.zeros([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), self.testImages.shape[0]])
		# for i in range(self.testImages.shape[0]):
		# 	batchTestData = self.testImages[i, :].reshape(1, 3*self.imgSize[0]*self.imgSize[1])
		# 	batchTestLabels = self.testPoints[i, :].reshape(1, self.num_landmarks*2)

		# 	temp = self.sess.run([self.out_feature],
		# 		feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
		# 	# print(np.array(temp).shape)
		# 	temp = np.array(temp)
		# 	X_features[:, i]= temp[0,0,:]
		# np.save('../experiments/CIFARCase1/Features/features_test.npy', X_features)

	def PCA_test_model(self):
		
		'''
		Computes the PCA of the network feature space

		'''

		self.saver = tf.train.Saver()
		self.load(self.checkpoint_dir, self.model_dir, self.model_name)
		numE = self.orig_data_test.shape[0]
		print(numE)   
		allaccuracy = 0
		X_features = np.zeros([2*2*512, numE])
		Y_features = np.zeros([2*2*512, numE])
		for i in range(numE):
			batchTestData = self.orig_data_test[i, :].reshape(1, 3*32*32)
			batchTestLabels = self.orig_labels_test[i, :].reshape(1, self.numL)

			temp = self.sess.run([self.out_feature],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
			# print(np.array(temp).shape)
			temp = np.array(temp)
			X_features[:, i]= temp[0,0,:]
		np.save('../experiments/CIFARCase1/Features/features_test.npy', X_features)
		meanF = np.mean(X_features, 1)
		for i in range(numE):
			Y_features[..., i] = X_features[..., i] - meanF

		trickCovMat = np.dot(Y_features.T,Y_features) * 1.0/np.sqrt(numE-1)
		[s,v] = np.linalg.eigh(trickCovMat)
		eigs=s[::-1]
		y = np.cumsum(eigs) / np.sum(eigs)
		plt.figure()
		plt.plot(y)
		plt.savefig('Screeplot-CIFAR-Base-50.png')


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

class Landmark_Case5_Classification:

	'''
	This class defines the state of the art CIFAR 10 classifier.
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
		self.num_landmarks = int(ut.readSpecificTagXML(root, 'num_landmarks'))
		self.imgSize = np.array(ut.readSpecificTagXML(root, 'input_dims').split(" "), dtype=np.int64)
		# self.data_path = readSpecificTagXML(root, 'data_path')
		self.landmarkImages = ut.readSpecificTagXML(root, 'landmark_images')
		self.landmarkPoints = ut.readSpecificTagXML(root, 'landmark_points')
		self.bottleneck = int(ut.readSpecificTagXML(root, 'bottleneck'))
		self.noise_var = float(ut.readSpecificTagXML(root, 'noise_var'))
		self.cae_weight_init = float(ut.readSpecificTagXML(root, 'cae_weight_init'))
		self.cae_weight_final = float(ut.readSpecificTagXML(root, 'cae_weight_final')) 
		self.burn_in_epochs = int(ut.readSpecificTagXML(root, 'burn_in'))
		# read and process the data sequence
		data_all = ut.landmarkDataPartition(self.landmarkImages, self.landmarkPoints, self.imgSize, self.num_landmarks)
		self.trainImages = data_all.trainImages(1)
		self.trainPoints = data_all.trainPoints(1)
		self.valImages = data_all.valImages(1)
		self.valPoints = data_all.valPoints(1)
		self.testImages = data_all.testImages(1)
		self.testPoints = data_all.testPoints(1)
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		self.logFN = self.log_dir + '/logFile.csv'
		self.num_batches = int(self.trainImages.shape[0] / self.batch_size)

	def feature_extractor(self, x, phase_train, is_training=True, reuse=False):

		''' 
		The initial network which represents the feature extraction
		part of deep network

		'''
		with tf.variable_scope("feature_extractor", reuse=reuse):
			x_in = tf.reshape(x, [-1, self.imgSize[0], self.imgSize[1], 3])

			conv1_1 = ut.conv_layer(x_in, 3, 8, 'conv1_1', 3, phase_train)
			# conv1_2 = ut.conv_layer(conv1_1, 64, 128, 'conv1_2', 3, phase_train)
			with tf.name_scope('pool1'):
				h_pool1 = ut.max_pool_2x2_2d(conv1_1)
			# conv2_1 = ut.conv_layer(h_pool1, 128, 256, 'conv2_1', 3, phase_train)
			conv2_1 = ut.conv_layer(h_pool1, 8, 16, 'conv2_1', 3, phase_train)
			with tf.name_scope('pool2'):
				h_pool2 = ut.max_pool_2x2_2d(conv2_1)
			conv3_1 = ut.conv_layer(h_pool2, 16, 32, 'conv3_1', 5, phase_train)
			with tf.name_scope('pool3'):
				h_pool3 = ut.max_pool_2x2_2d(conv3_1)
			conv4_1 = ut.conv_layer(h_pool3, 32, 64, 'conv4_1', 5, phase_train)
			with tf.name_scope('pool4'):
				h_pool4 = ut.max_pool_2x2_2d(conv4_1)
			conv5_1 = ut.conv_layer(h_pool4, 64, 128, 'conv5_1', 5, phase_train)
			with tf.name_scope('pool5'):
				h_pool5 = ut.max_pool_2x2_2d(conv5_1)
				h_pool5_flat = tf.reshape(h_pool5, [-1, int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0))])
			return h_pool5_flat

	def classifier(self, x, phase_train, is_training=True, reuse=False):

		'''
		The end part of the deep network which takes in the extracted
		features and classifies them into digits

		'''
		with tf.variable_scope('classifier', reuse=reuse):
			with tf.name_scope('fc1'):
				W_fc1 = ut.weight_variable([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), 256], 'fc1_wt')
				b_fc1 = ut.bias_variable([256], 'fc1_bias')
				h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
			# with tf.name_scope('fc2'):
			# 	W_fc2 = ut.weight_variable([128, 256], 'fc2_wt')
			# 	b_fc2 = ut.bias_variable([256], 'fc2_bias')
			# 	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
			# with tf.name_scope('fc3'):
			# 	W_fc3 = ut.weight_variable([256, 512], 'fc3_wt')
			# 	b_fc3 = ut.bias_variable([512], 'fc3_bias')
			# 	h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
			# with tf.name_scope('fc4'):
			# 	W_fc4 = ut.weight_variable([512, 1024], 'fc4_wt')
			# 	b_fc4 = ut.bias_variable([1024], 'fc4_bias')
			# 	h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)
			with tf.name_scope('fc5'):
				W_fc5 = ut.weight_variable([256, self.num_landmarks*2], 'fc5_wt')
				b_fc5 = ut.bias_variable([self.num_landmarks*2], 'fc5_bias')
				out = tf.matmul(h_fc1, W_fc5) + b_fc5
			return out

	def CAE(self, x, phase_train, is_training=True, reuse=False):
		with tf.variable_scope('CAE', reuse=reuse):
			with tf.name_scope('fc1-cae'):
				W_fc1_cae = ut.weight_variable([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0)), 256], 'fc1_wt_cae')
				b_fc1_cae = ut.bias_variable([256], 'fc1_bias_cae')
				h_fc1_cae = ut.parametric_relu(tf.matmul(x, W_fc1_cae) + b_fc1_cae)
			with tf.name_scope('fc1b-cae'):
				W_fc1b_cae = ut.weight_variable([256, 128], 'fc1b_wt_cae')
				b_fc1b_cae = ut.bias_variable([128], 'fc1b_bias_cae')
				h_fc1b_cae = ut.parametric_relu(tf.matmul(h_fc1_cae, W_fc1b_cae) + b_fc1b_cae)
			# with tf.name_scope('fc1c-cae'):
			# 	W_fc1c_cae = ut.weight_variable([512, 256], 'fc1c_wt_cae')
			# 	b_fc1c_cae = ut.bias_variable([256], 'fc1c_bias_cae')
			# 	h_fc1c_cae = ut.parametric_relu(tf.matmul(h_fc1b_cae, W_fc1c_cae) + b_fc1c_cae)
			with tf.name_scope('fc2-cae'):
				W_fc2_cae = ut.weight_variable([128, self.bottleneck], 'fc2_wt_cae')
				b_fc2_cae = ut.bias_variable([self.bottleneck], 'fc2_bias_cae')
				h_fc2_cae = ut.parametric_relu(tf.matmul(h_fc1b_cae, W_fc2_cae) + b_fc2_cae)
			with tf.name_scope('fc3-cae'):
				W_fc3_cae = ut.weight_variable([self.bottleneck, 128], 'fc3_wt_cae')
				b_fc3_cae = ut.bias_variable([128], 'fc3_bias_cae')
				h_fc3_cae = ut.parametric_relu(tf.matmul(h_fc2_cae, W_fc3_cae) + b_fc3_cae)
			# with tf.name_scope('fc3c-cae'):
			# 	W_fc3c_cae = ut.weight_variable([256, 512], 'fc3c_wt_cae')
			# 	b_fc3c_cae = ut.bias_variable([512], 'fc3c_bias_cae')
			# 	h_fc3c_cae = ut.parametric_relu(tf.matmul(h_fc3_cae, W_fc3c_cae) + b_fc3c_cae)
			with tf.name_scope('fc3b-cae'):
				W_fc3b_cae = ut.weight_variable([128, 256], 'fc3b_wt_cae')
				b_fc3b_cae = ut.bias_variable([256], 'fc3b_bias_cae')
				h_fc3b_cae = ut.parametric_relu(tf.matmul(h_fc3_cae, W_fc3b_cae) + b_fc3b_cae)
			with tf.name_scope('fc4-cae'):
				W_fc4_cae = ut.weight_variable([256, int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0))], 'fc4_wt_cae')
				b_fc4_cae = ut.bias_variable([int((self.imgSize[0]*self.imgSize[1]*128)/(32.0*32.0))], 'fc4_bias_cae')
				h_fc4_cae = ut.parametric_relu(tf.matmul(h_fc3b_cae, W_fc4_cae) + b_fc4_cae)
			return  [h_fc4_cae, h_fc2_cae]

	def build_model(self):

		'''
		Builds the graph of the network, adds in the loss functions,
		defines placeholders.

		'''
		self.input_data = tf.placeholder(tf.float32, [None, self.imgSize[0]*self.imgSize[1]*3], name='orig_data')
		self.labels = tf.placeholder(tf.float32, [None, self.num_landmarks*2], name='orig_labels')
		self.phase_train = tf.placeholder(tf.bool, name='phase')
		self.cae_weight = tf.placeholder(tf.float32, name='wt')
		self.emptyPH = tf.placeholder(tf.float32, [None, self.bottleneck], name='emptyPH')
		self.out_feature = self.feature_extractor(self.input_data, self.phase_train, is_training=True, reuse=False)
		self.out_labels = self.classifier(self.out_feature, self.phase_train, is_training=True, reuse=False)
		self.noisy_features = self.out_feature + self.noise_var * tf.random_normal(tf.shape(self.out_feature), 0, 1, dtype=tf.float32)
		[self.cae_out, self.bottleneck_value] = self.CAE(self.noisy_features, self.phase_train, is_training=True, reuse=False)
		self.cae_loss = tf.reduce_mean(tf.nn.l2_loss(self.cae_out - self.out_feature)/ tf.nn.l2_loss(self.out_feature))
		self.l1Loss = tf.reduce_mean(tf.losses.absolute_difference(self.bottleneck_value, self.emptyPH))
		# define the cross entropy loss
		# self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_labels, labels=self.labels))
		# correct_pred = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.out_labels, 1))
		# self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
		self.squaredLoss = tf.reduce_mean((self.out_labels - self.labels)**2)
		self.loss = self.squaredLoss + self.cae_weight*self.cae_loss + self.l1_wt*self.l1Loss
		'''Training function'''
		t_vars = tf.trainable_variables()
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optim_burnin = tf.train.AdamOptimizer(self.learning_rate).minimize(self.squaredLoss, var_list=t_vars)
			self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=t_vars)

	def train_model(self):

		''' 
		Implements the training routineself.

		'''
		print("Here")
		self.saver = tf.train.Saver()
		print("Here")
		self.sess.run(tf.global_variables_initializer())
		print("Here")
		all_train_loss = []
		all_train_acc = []
		all_val_loss = []
		all_val_acc = []
		start_epoch = 0
		start_batch_id = 0
		counter = 1
		start_time = time.time()
		
		with open(self.logFN, 'w') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',')
			for epoch in range(start_epoch, self.epochs):
				[shufData, shufLables] = ut.training_shuffle_data(self.trainImages, self.trainPoints)
				total_accuracy_train = 0
				total_loss_train = 0
				total_accuracy_val = 0
				total_loss_val = 0
				total_l1_train = 0
				caeWT = self.cae_weight_init + (epoch/self.epochs)*(self.cae_weight_final - self.cae_weight_init)
				for idx in range(start_batch_id, self.num_batches):
					[batch_data_train, batch_labels_train] = ut.get_batch_data(shufData, shufLables, idx, self.batch_size)
					batch_emptyPH_tr = np.zeros([batch_labels_train.shape[0], self.bottleneck])
					# [batch_data_val, batch_labels_val] = ut.get_batch_data(self.valImages, self.valPoints, idx, self.batch_size)
					if epoch > self.burn_in_epochs:
						self.optim.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True, self.cae_weight : caeWT, self.emptyPH:batch_emptyPH_tr} )
					else:
						self.optim_burnin.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True, self.cae_weight : caeWT, self.emptyPH:batch_emptyPH_tr} )

					counter += 1
					# training accuracy

					if np.mod(counter, self.print_iter) == 0:
						# validation accuracy
						accuracy_train, match_train, cae_train, l1_train = self.sess.run([self.loss, self.squaredLoss, self.cae_loss, self.l1Loss],
							feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:False, self.cae_weight : caeWT, self.emptyPH:batch_emptyPH_tr})

						# accuracy_val, match_val, cae_val, featureNV, tot_val = self.sess.run([self.accuracy, self.cross_entropy, self.cae_loss, self.featurenorm, self.loss],
						# 	feed_dict={self.input_data: batch_data_val, self.labels: batch_labels_val, self.phase_train:False, self.cae_weight : caeWT})	
						# l = [epoch, idx, accuracy_train, accuracy_val, match_train, match_val, cae_train, cae_val, featureNT, featureNV]
						# spamwriter.writerow(l)
						print("TRAINING : Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, Match-Loss: %.8f, CAE-Loss: %.8f, l1-Loss : %.8f" %(epoch, idx, self.num_batches, time.time() - start_time, accuracy_train, match_train, cae_train, l1_train))

				if np.mod(epoch, self.save_iter) == 0:
					# save the check point
					self.save(self.checkpoint_dir, self.model_dir, self.model_name, counter)

	def test_model(self):

		'''
		Implements the testing routine.

		'''

		self.saver = tf.train.Saver()
		self.load(self.checkpoint_dir, self.model_dir, self.model_name)
		loadBT = 1
		numE = int(self.testImages.shape[0]/loadBT)
		allaccuracy = 0
		allaccuracy_cae = 0
		allaccuracy_squared = 0
		np.save(os.path.join(self.result_dir, self.model_name) + 'testImages_CAE.npy', self.testImages)
		np.save(os.path.join(self.result_dir, self.model_name) + 'testPoints_CAE.npy', self.testPoints)
		outTestPoints = np.zeros([self.testPoints.shape[0], self.testPoints.shape[1]])
		for i in range(numE):
			batchTestData = self.testImages[i*loadBT:(i+1)*loadBT, :]
			batchTestLabels = self.testPoints[i*loadBT:(i+1)*loadBT, :]
			batch_emptyPH_test = np.zeros([batchTestLabels.shape[0], self.bottleneck])
			accuracy_test, squared_test, out_test, orig_feature, cae_feature = self.sess.run([self.loss, self.squaredLoss, self.out_labels, self.out_feature, self.cae_out],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False, self.cae_weight:self.cae_weight_init, self.emptyPH: batch_emptyPH_test})
			allaccuracy += accuracy_test
			allaccuracy_squared += squared_test
			outTestPoints[i, ...] = out_test
			temp = np.sqrt(np.sum((orig_feature - cae_feature)**2))
			tempden = np.sqrt(np.sum(orig_feature**2))
			print(temp, tempden)
			allaccuracy_cae += temp / tempden
		np.save(os.path.join(self.result_dir, self.model_name) + 'outputPoints_CAE.npy', outTestPoints)
		allaccuracy = allaccuracy / numE
		allaccuracy_cae = allaccuracy_cae / numE
		allaccuracy_squared = allaccuracy_squared / numE
		print("TESTING ACCURACY: %.8f,  squaredLoss: %.8f, cae-LOss : %.8f" % (allaccuracy, allaccuracy_squared, allaccuracy_cae))
		return allaccuracy

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
