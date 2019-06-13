##########################################################################
#																		 #
# Desc : Training for Reduced MNIST classification with different        #
# 		 architectures and training paradigms							 #
# Author : Riddhish Bhalodia											 #
# Institution : Scientific Computing and Imaging Institute				 #
# Date : 18th December 2018												 #
#																		 #
##########################################################################

import numpy as np
import tensorflow as tf
import csv
import xml.etree.ElementTree as ET
import sys


sys.path.append('../ext')
import utils
import networks
from utils import show_all_variables
from utils import check_folder

xmlfile='../xmls/case4MNIST.xml'
# first read the xml file
tree=ET.parse(xmlfile)
root = tree.getroot()
# get the case number 
caseNO = int(utils.readSpecificTagXML(root, 'case_number'))

print('....................................')
print('Working with Case Number : ', caseNO)
print('....................................')

if caseNO==1:
	# load the graph first
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.5
	sess = tf.Session(config=config)
	net = networks.MNIST_Base_Classification(sess, xmlfilename=xmlfile)

	# build the graph
	net.build_model()

	show_all_variables()

	net.train_model()

	net.test_model()
	# net.PCA_test_model()

if caseNO==2:
	# load the graph first
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.5
	sess = tf.Session(config=config)
	net = networks.MNIST_Case2_Classification(sess, xmlfilename=xmlfile)

	# build the graph
	net.build_model()

	show_all_variables()

	net.train_model()

	net.test_model()

if caseNO==3:
	# load the graph first
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.5
	sess = tf.Session(config=config)
	net = networks.MNIST_Case3_Classification(sess, xmlfilename=xmlfile)

	# build the graph
	net.build_model()

	show_all_variables()

	net.train_model()

	net.test_model()

if caseNO==4:
	
	config = tf.ConfigProto()
	# config.gpu_options.allow_growth = True
	# config.gpu_options.per_process_gpu_memory_fraction = 0.5
	sess = tf.Session(config=config)
	net = networks.MNIST_Case4_Classification(sess, xmlfilename=xmlfile)
	# build the graph
	net.build_model()
	show_all_variables()
	net.train_model()
	net.test_model()
	net.validation_model()