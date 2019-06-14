import numpy as np
import tensorflow as tf
import csv
import xml.etree.ElementTree as ET
import sys
from argparse import ArgumentParser
sys.path.append('../ext')
import utils
import networks_Toy
from utils import show_all_variables
from utils import check_folder

def temp(xmlfile, caseID):

	if caseID == "CAE":

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.75
		sess = tf.Session(config=config)
		net = networks_Toy.Toy_CAE_Only(sess, xmlfilename=xmlfile)
		net.build_model()
		net.train_model()
	
	if caseID == "Hardcon":
		# load the graph first
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.75
		sess = tf.Session(config=config)
		net = networks_Toy.Toy_Hard(sess, xmlfilename=xmlfile)
		net.build_model()
		net.train_model()
	# step = 
	# [outLab, outBT] = net.VisualizeModel(step)

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--xmlfilepath", type=float, dest="xmlfilename", help="xml file path containing parameters")
	parser.add_argument("--caseID", type=float, dest="caseID", default="CAE", help="the case ID for experimentation")
	args = parser.parse_args()
	temp(**vars(args))
