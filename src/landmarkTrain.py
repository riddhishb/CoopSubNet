import numpy as np
import tensorflow as tf
import csv
import xml.etree.ElementTree as ET
import sys
sys.path.append('../ext')
import utils
import networks_landmarks
from argparse import ArgumentParser
from utils import show_all_variables
from utils import check_folder


def temp(xmlfile, caseID):
	# first read the xml file

	print('....................................')
	print('Working with Case : ', caseID)
	print('....................................')

	if caseID=="CAE":
		# load the graph first
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.5
		sess = tf.Session(config=config)
		net = networks_landmarks.Landmark_CAE_Classification(sess, xmlfilename=xmlfile)

		# build the graph
		net.build_model()

		show_all_variables()

		net.train_model()
		# net.validation_model()
		# net.test_model()

	if caseID=="Hardcon":
		# load the graph first
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.5
		sess = tf.Session(config=config)
		net = networks_landmarks.MNIST_Hardcon_Classification(sess, xmlfilename=xmlfile)

		# build the graph
		net.build_model()

		show_all_variables()

		net.train_model()
		# net.validation_model()
		# net.test_model()

	if caseID=="CAEL1":
		# load the graph first
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.5
		sess = tf.Session(config=config)
		net = networks_landmarks.Landmark_CAEL1Norm_Classification(sess, xmlfilename=xmlfile)

		# build the graph
		net.build_model()

		show_all_variables()

		net.train_model()
		# net.validation_model()
		# net.test_model()

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--xmlfilepath", type=float, dest="xmlfile", help="xml file path containing parameters")
	parser.add_argument("--caseID", type=float, dest="caseID", default="CAE", help="the case ID for experimentation")
	args = parser.parse_args()
	temp(**vars(args))

