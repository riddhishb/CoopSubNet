##########################################################################
#																		 #
# Desc : All semi supervised for cooperative networks 			 #
# Author : Riddhish Bhalodia											 #
# Institution : Scientific Computing and Imaging Institute				 #
# Date : 27th February 2019												 #
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
import io
import tfmpl
import matplotlib.pyplot as plt

class Toy_Hard:

	'''
	This class defines the state of the art MNIST classifier.
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
		# self.data_path = readSpecificTagXML(root, 'data_path')
		self.reduced_data = int(ut.readSpecificTagXML(root, 'reduced_data')) # number of data per label to use for training
		
		# read and process the data sequence
		self.dataAll = np.load('../data/ToyData/Data.npy')
		self.labelsAll = np.load('../data/ToyData/labels.npy')
		[self.orig_data, self.orig_labels, self.orig_data_val, self.orig_labels_val, self.orig_data_test, self.orig_labels_test] = ut.partitionToy(self.dataAll, self.labelsAll, self.reduced_data)
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		self.logFN = self.log_dir + '/logFile_' + '_' + str(self.batch_size) + '.csv'
		self.num_batches = int(self.orig_data.shape[0] / self.batch_size)

	def feature_extractor(self, x, phase_train, is_training=True, reuse=False):

		''' 
		The initial network which represents the feature extraction
		part of deep network

		'''
		with tf.variable_scope("feature_extractor", reuse=reuse):
			with tf.name_scope('fc1'):
				W_fc1 = ut.weight_variable([500, 256], 'fc1_wt')
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
			
			return h_fc3


	def classifier(self, x, phase_train, is_training=True, reuse=False):

		'''
		The end part of the deep network which takes in the extracted
		features and classifies them into digits

		'''
		with tf.variable_scope('classifier', reuse=reuse):
			with tf.name_scope('fc4'):
				W_fc2 = ut.weight_variable([256, 128], 'fc4_wt')
				b_fc2 = ut.bias_variable([128], 'fc4_bias')	
				h_fc2 = tf.nn.relu(tf.matmul(x, W_fc2) + b_fc2)

			with tf.name_scope('fc5'):
				W_fc3 = ut.weight_variable([128, 4], 'fc5_wt')
				b_fc3 = ut.bias_variable([4], 'fc5 _bias')
				out = tf.matmul(h_fc2, W_fc3) + b_fc3
			
			return out

	def build_model(self):

		'''
		Builds the graph of the network, adds in the loss functions,
		defines placeholders.

		'''
		self.input_data = tf.placeholder(tf.float32, [None, 500], name='orig_data')
		self.labels = tf.placeholder(tf.float32, [None, 4], name='orig_labels')
		self.phase_train = tf.placeholder(tf.bool, name='phase')

		self.out_feature = self.feature_extractor(self.input_data, self.phase_train, is_training=True, reuse=False)
		self.out_labels = self.classifier(self.out_feature, self.phase_train, is_training=True, reuse=False)

		# define the cross entropy loss
		self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_labels, labels=self.labels))
		correct_pred = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.out_labels, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
		self.loss = self.cross_entropy
		'''Training function'''
		t_vars = tf.trainable_variables()
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy, var_list=t_vars)

	def train_model(self):

		''' 
		Implements the training routineself.

		'''
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
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
				[shufData, shufLables] = ut.training_shuffle_data(self.orig_data, self.orig_labels)
				total_accuracy_train = 0
				total_loss_train = 0
				total_accuracy_val = 0
				total_loss_val = 0

				for idx in range(start_batch_id, self.num_batches):
					[batch_data_train, batch_labels_train] = ut.get_batch_data(shufData, shufLables, idx, self.batch_size)
					[batch_data_val, batch_labels_val] = ut.get_batch_data(self.orig_data_val, self.orig_labels_val, idx, self.batch_size)
					self.optim.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True} )

					counter += 1
					accuracy_train, out_train, match_train = self.sess.run([self.accuracy, self.out_labels, self.cross_entropy],
							feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:False})

				[acc, mtLs, Ls] = self.all_model_loss(shufData, shufLables)
				[acc_val, mtLs_val, Ls_val] = self.all_model_loss(self.orig_data_val, self.orig_labels_val)	
				l = [epoch, acc, acc_val, mtLs, mtLs_val, Ls, Ls_val]
				spamwriter.writerow(l)
				print("-----------------------------")
				print("Training-Labeled: Epoch: [%2d], accuracy: %.8f, Match-Loss: %.8f, Tot-Loss: %.8f" %(epoch, acc, mtLs, Ls))
				print("Validation: Epoch: [%2d], accuracy: %.8f, Match-Loss: %.8f, Tot-Loss: %.8f" %(epoch, acc_val, mtLs_val, Ls_val))
				print("-----------------------------")
				if np.mod(epoch, self.save_iter) == 0:
					self.save_separate(self.checkpoint_dir, self.model_dir, self.model_name, self.reduced_data, counter)

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

	def all_model_loss(self, data, labels):
		'''
		Implements the testing routine.
		Also computes the CAE loss for reconstruction accuracy

		'''

		# self.saver = tf.train.Saver()
		# self.load_separate_step(self.checkpoint_dir, self.model_dir, self.model_name, self.bottleneck, self.reduced_data, step)
		loadBT = 4
		numE = int(data.shape[0]/loadBT)
		allaccuracy = 0
		allmatchloss = 0
		allloss = 0
		for i in range(numE):
			batchTestData = data[i*loadBT:(i+1)*loadBT, :]
			batchTestLabels = labels[i*loadBT:(i+1)*loadBT, :]
			masks = np.ones([batchTestData.shape[0], 4])
			accuracy_test, match_loss, loss = self.sess.run([self.accuracy, self.cross_entropy, self.loss],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False, })
			allaccuracy += accuracy_test
			allmatchloss += match_loss
			allloss += loss

		allaccuracy = allaccuracy / numE
		allloss = allloss / numE
		allmatchloss = allmatchloss / numE
		return [allaccuracy, allmatchloss, allloss]


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

	def save_separate(self, checkpoint_dir, model_dir, model_name, rd, step):
		model_name = model_name + '_' + str(rd)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir, model_name)
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name + '.model'), global_step=step)

	def load_separate(self, checkpoint_dir, model_dir, model_name, rd):
		model_name = model_name  + '_' + str(rd)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir, model_name)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		print('Loading Checkpoint ...')
		print(ckpt_name)
		self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))


class Toy_CAE_Only:

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
		self.reduced_data = int(ut.readSpecificTagXML(root, 'reduced_data')) # number of data per label to use for training
		self.bottleneck = int(ut.readSpecificTagXML(root, 'bottleneck'))
		print('BOTTLENECK = ', self.bottleneck)
		self.cae_weight = float(ut.readSpecificTagXML(root, 'cae_weight'))
		self.burn_in_epochs = int(ut.readSpecificTagXML(root, 'burn_in'))
		# read and process the data sequence
		self.dataAll = np.load('../data/ToyData/Data.npy')
		self.labelsAll = np.load('../data/ToyData/labels.npy')
		print(np.argmax(self.labelsAll, 1).shape)
		[self.orig_data, self.orig_labels, self.orig_data_val, self.orig_labels_val, self.orig_data_test, self.orig_labels_test] = ut.partitionToy(self.dataAll, self.labelsAll, self.reduced_data)
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		self.logFN = self.log_dir + '/logFile_' + str(self.bottleneck) + '_' + str(self.batch_size) + '_' + str(self.cae_weight_init) + '.csv'
		self.TBLog = self.log_dir + '/' + str(self.bottleneck) + '_' + str(self.batch_size) + '_' + str(self.cae_weight_init) + '/log4'
		
		self.num_batches = int(self.orig_data.shape[0] / self.batch_size)

	def feature_extractor(self, x, phase_train, is_training=True, reuse=False):

		''' 
		The initial network which represents the feature extraction
		part of deep network

		'''
		with tf.variable_scope("feature_extractor", reuse=reuse):
			with tf.name_scope('fc1'):
				W_fc1 = ut.weight_variable([500, 256], 'fc1_wt')
				b_fc1 = ut.bias_variable([256], 'fc1_bias')	
				h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
			
			return h_fc1


	def classifier(self, x, phase_train, is_training=True, reuse=False):

		'''
		The end part of the deep network which takes in the extracted
		features and classifies them into digits

		'''
		with tf.variable_scope('classifier', reuse=reuse):
			with tf.name_scope('fc2'):
				W_fc2 = ut.weight_variable([256, 128], 'fc2_wt')
				b_fc2 = ut.bias_variable([128], 'fc2_bias')	
				h_fc2 = tf.nn.relu(tf.matmul(x, W_fc2) + b_fc2)

			with tf.name_scope('fc3'):
				W_fc3 = ut.weight_variable([128, 4], 'fc3_wt')
				b_fc3 = ut.bias_variable([4], 'fc3_bias')
				out = tf.matmul(h_fc2, W_fc3) + b_fc3
			
			return out

	def CAE(self, x, phase_train, is_training=True, reuse=False):

		'''
		The external autoencoder based regualrization
		This is the Cooperative Autoencoder (CAE)
		'''

		with tf.variable_scope('CAE', reuse=reuse):
			with tf.name_scope('fc1-cae'):
				W_fc1_cae = ut.weight_variable([256, 128], 'fc1_wt_cae')
				b_fc1_cae = ut.bias_variable([128], 'fc1_bias_cae')
				h_fc1_cae = ut.parametric_relu(tf.matmul(x, W_fc1_cae) + b_fc1_cae)
			with tf.name_scope('fc1b-cae'):
				W_fc1b_cae = ut.weight_variable([128, 64], 'fc1b_wt_cae')
				b_fc1b_cae = ut.bias_variable([64], 'fc1b_bias_cae')
				h_fc1b_cae = ut.parametric_relu(tf.matmul(h_fc1_cae, W_fc1b_cae) + b_fc1b_cae)
			with tf.name_scope('fc2b-cae'):
				W_fc2b_cae = ut.weight_variable([64, self.bottleneck], 'fc2b_wt_cae')
				b_fc2b_cae = ut.bias_variable([self.bottleneck], 'fc2b_bias_cae')
				h_fc2b_cae = ut.parametric_relu(tf.matmul(h_fc1b_cae, W_fc2b_cae) + b_fc2b_cae)
			with tf.name_scope('fc3b-cae'):
				W_fc3b_cae = ut.weight_variable([self.bottleneck, 64], 'fc3b_wt_cae')
				b_fc3b_cae = ut.bias_variable([64], 'fc3b_bias_cae')
				h_fc3b_cae = ut.parametric_relu(tf.matmul(h_fc2b_cae, W_fc3b_cae) + b_fc3b_cae)
			with tf.name_scope('fc4b-cae'):
				W_fc4b_cae = ut.weight_variable([64, 128], 'fc4b_wt_cae')
				b_fc4b_cae = ut.bias_variable([128], 'fc4b_bias_cae')
				h_fc4b_cae = ut.parametric_relu(tf.matmul(h_fc3b_cae, W_fc4b_cae) + b_fc4b_cae)
			with tf.name_scope('fc4-cae'):
				W_fc4_cae = ut.weight_variable([128, 256], 'fc4_wt_cae')
				b_fc4_cae = ut.bias_variable([256], 'fc4_bias_cae')
				h_fc4_cae = ut.parametric_relu(tf.matmul(h_fc4b_cae, W_fc4_cae) + b_fc4_cae)

			return [h_fc4_cae, h_fc2b_cae]

	def build_model(self):

		'''
		Builds the graph of the network, adds in the loss functions,
		defines placeholders.

		'''
		self.input_data = tf.placeholder(tf.float32, [None, 500], name='orig_data')
		self.labels = tf.placeholder(tf.float32, [None, 4], name='orig_labels')
		self.phase_train = tf.placeholder(tf.bool, name='phase')
		self.out_feature = self.feature_extractor(self.input_data, self.phase_train, is_training=True, reuse=False)
		self.out_labels = self.classifier(self.out_feature, self.phase_train, is_training=True, reuse=False)
		self.out_labels_trv = tf.argmax(tf.nn.softmax(self.out_labels), 1)
		[self.cae_out, self.bottleneck_value] = self.CAE(self.out_feature, self.phase_train, is_training=True, reuse=False)
		self.out_labels_fromCAE = self.classifier(self.cae_out, False, is_training=False, reuse=True)
		self.cae_loss = tf.reduce_mean(tf.nn.l2_loss(self.cae_out - self.out_feature)/ tf.nn.l2_loss(self.out_feature))
		self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_labels, labels=self.labels))
		correct_pred = tf.equal(tf.argmax(self.labels, 1), tf.argmax(tf.nn.softmax(self.out_labels), 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
		correct_pred_CAE = tf.equal(tf.argmax(self.labels, 1), tf.argmax(tf.nn.softmax(self.out_labels_fromCAE), 1))
		self.accuracy_CAE = tf.reduce_mean(tf.cast(correct_pred_CAE, tf.float32), name='accuracy-CAE')

		self.loss = self.cross_entropy + self.cae_weight*self.cae_loss

		'''Training function'''
		t_vars = tf.trainable_variables()
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optim_burnin = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy, var_list=t_vars)
			self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=t_vars)
			# self.optimall = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=t_vars)

		with tf.name_scope('performance-evaluation'):
			self.tb_acc = tf.placeholder(tf.float32, shape=None, name='acc-summary')
			self.tb_ce = tf.placeholder(tf.float32, shape=None, name='cross-entropy-summary')
			self.tb_cae = tf.placeholder(tf.float32, shape=None, name='cae-loss-summary')
			self.tb_tl = tf.placeholder(tf.float32, shape=None, name='total-loss-summary')
			self.tb_acc2 = tf.placeholder(tf.float32, shape=None, name='acc-summary2')
			self.tb_ce2 = tf.placeholder(tf.float32, shape=None, name='cross-entropy-summary2')
			self.tb_cae2 = tf.placeholder(tf.float32, shape=None, name='cae-loss-summary2')
			self.tb_tl2 = tf.placeholder(tf.float32, shape=None, name='total-loss-summary2')
			self.tb_acc_summ = tf.summary.scalar('accuracy-train', self.tb_acc)
			self.tb_ce_summ = tf.summary.scalar('cross-entropy-train', self.tb_ce)
			self.tb_cae_summ = tf.summary.scalar('cae-loss-train', self.tb_cae)
			self.tb_tl_summ = tf.summary.scalar('total-loss-train', self.tb_tl)
			self.tb_acc_summ2 = tf.summary.scalar('accuracy-val', self.tb_acc2)
			self.tb_ce_summ2 = tf.summary.scalar('cross-entropy-val', self.tb_ce2)
			self.tb_cae_summ2 = tf.summary.scalar('cae-loss-val', self.tb_cae2)
			self.tb_tl_summ2 = tf.summary.scalar('total-loss-val', self.tb_tl2)

		self.performance_summaries = tf.summary.merge_all()

	def train_model(self):

		''' 
		Implements the training routineself.

		'''
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		self.writer = tf.summary.FileWriter(self.TBLog, self.sess.graph)
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
					if epoch > self.burn_in_epochs:
						self.optim.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True} )
					else:
						self.optim_burnin.run(session=self.sess, feed_dict={self.input_data: batch_data_train, self.labels: batch_labels_train, self.phase_train:True} )

					counter += 1
					# training accuracy

				[acc, caeerr, mtLs, Ls] = self.all_model_loss(shufData, shufLables)
				[acc_val, caeerr_val, mtLs_val, Ls_val] = self.all_model_loss(self.orig_data_val, self.orig_labels_val)	
				
				# print("Dimension ", img.shape)
				summ_train = self.sess.run(self.performance_summaries, feed_dict={self.tb_acc:acc, self.tb_ce:mtLs, self.tb_cae:caeerr, self.tb_tl:Ls,self.tb_acc2:acc_val, self.tb_ce2:mtLs_val, self.tb_cae2:caeerr_val, self.tb_tl2:Ls_val})
				self.writer.add_summary(summ_train, epoch)
				
				l = [epoch, acc, acc_val, caeerr, caeerr_val, mtLs, mtLs_val, Ls, Ls_val]
				spamwriter.writerow(l)
				print("-----------------------------")
				print("Training: Epoch: [%2d], accuracy: %.8f, Match-Loss: %.8f, CAE-Err: %.8f, Tot-Loss: %.8f" %(epoch, acc, mtLs, caeerr, Ls))
				print("Validation: Epoch: [%2d], accuracy: %.8f, Match-Loss: %.8f, CAE-Err: %.8f, Tot-Loss: %.8f" %(epoch, acc_val, mtLs_val, caeerr_val, Ls_val))
				print("-----------------------------")
				if np.mod(epoch, 20) == 0:
					[outlab, outval] = self.VisualizeModel()
					bufimg = self.getScatter(outlab, outval, np.argmax(self.labelsAll, 1), 'BN values clustered by predicted Labels', 'BN values clustered by true Labels')
					img = tf.image.decode_png(bufimg.getvalue(), channels = 4)
					img = tf.expand_dims(img, 0)
					summ_img = tf.summary.image("Scatter-pred-v-BT", img, max_outputs=1)
					self.writer.add_summary(summ_img.eval(session=self.sess), epoch)
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

	def VisualizeModel(self):
		# self.saver = tf.train.Saver()
		# self.load_separate_step(self.checkpoint_dir, self.model_dir, self.model_name, self.bottleneck, self.reduced_data, step)
		loadBT = 1
		numE = int(self.dataAll.shape[0]/loadBT)
		allaccuracy = 0
		allaccuracyCAE = 0
		cae_err = 0
		outputLabels = np.zeros([self.labelsAll.shape[0], 1])
		outputBTVal = np.zeros([self.labelsAll.shape[0], self.bottleneck])
		for i in range(numE):
			batchTestData = self.dataAll[i*loadBT:(i+1)*loadBT, :]
			batchTestLabels = self.labelsAll[i*loadBT:(i+1)*loadBT, :]

			labPred, btVal  = self.sess.run([self.out_labels_trv, self.bottleneck_value],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
			
			outputLabels[i, ...] = labPred
			outputBTVal[i, ...] = btVal

		return [outputLabels, outputBTVal]

	def getScatter(self, labels, Val, trueLabels, ttl1, ttl2):
		plt.figure()
		plt.subplot(121)
		for i in range(4):
			f = np.where(labels[:, 0] == i)
			f = f[0]
			# print(f.shape)
			plt.scatter(Val[f, 0], Val[f, 1])
		plt.title(ttl1)
		plt.subplot(122)
		for i in range(4):
			f = np.where(trueLabels == i)
			f = f[0]
			# print(f.shape)
			plt.scatter(Val[f, 0], Val[f, 1])
		plt.title(ttl2)
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)
		return buf

	def all_model_loss(self, data, labels):
		'''
		Implements the testing routine.
		Also computes the CAE loss for reconstruction accuracy

		'''
		loadBT = 4
		numE = int(data.shape[0]/loadBT)
		allaccuracy = 0
		cae_err = 0
		allmatchloss = 0
		allloss = 0
		for i in range(numE):
			batchTestData = data[i*loadBT:(i+1)*loadBT, :]
			batchTestLabels = labels[i*loadBT:(i+1)*loadBT, :]
			accuracy_test, orig_feature, cae_feature, match_loss, loss = self.sess.run([self.accuracy, self.out_feature, self.cae_out, self.cross_entropy, self.loss],
				feed_dict={self.input_data: batchTestData, self.labels: batchTestLabels, self.phase_train:False})
			allaccuracy += accuracy_test
			allmatchloss += match_loss
			allloss += loss
			temp = np.sqrt(np.sum((orig_feature - cae_feature)**2))
			tempden = np.sqrt(np.sum(orig_feature**2))
			cae_err += temp / tempden

		allaccuracy = allaccuracy / numE
		allloss = allloss / numE
		allmatchloss = allmatchloss / numE
		cae_err = cae_err / numE
		return [allaccuracy, cae_err, allmatchloss, allloss]

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
