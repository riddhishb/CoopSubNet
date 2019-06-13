import numpy as np
import tensorflow as tf
import csv
import xml.etree.ElementTree as ET
import sys
from argparse import ArgumentParser
sys.path.append('../ext')
import utils
import networks_Toy_SS
from utils import show_all_variables
from utils import check_folder

def temp(w, N, caseNo):
	if caseNo == 1:
		xmlfile='../xmls/Toy.xml'
		# first read the xml file
		tree=ET.parse(xmlfile)
		root = tree.getroot()
		# load the graph first
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.75
		sess = tf.Session(config=config)
		# print("For CAE weight : ", w)
		# utils.modifyXML(xmlfile, 'cae_weight_init', w)
		# utils.modifyXML(xmlfile, 'cae_weight_final', w)
		utils.modifyXML(xmlfile, 'num_epochs', int(N))
		net = networks_Toy_SS.Toy_Base(sess, xmlfilename=xmlfile)
		net.build_model()
		net.train_model()

	if caseNo == 2:
		xmlfile='../xmls/Toy_Cae.xml'
		# first read the xml file
		tree=ET.parse(xmlfile)
		root = tree.getroot()
		# load the graph first
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.75
		sess = tf.Session(config=config)
		print("For CAE weight : ", w)
		utils.modifyXML(xmlfile, 'cae_weight_init', w)
		utils.modifyXML(xmlfile, 'cae_weight_final', w)
		utils.modifyXML(xmlfile, 'num_epochs', int(N))
		net = networks_Toy_SS.Toy_CAE_Only(sess, xmlfilename=xmlfile)
		net.build_model()
		net.train_model()

	if caseNo == 3:
		xmlfile='../xmls/Toy_SS.xml'
		# first read the xml file
		tree=ET.parse(xmlfile)
		root = tree.getroot()
		# load the graph first
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.75
		sess = tf.Session(config=config)
		print("For CAE weight : ", w)
		utils.modifyXML(xmlfile, 'cae_weight_init', w)
		utils.modifyXML(xmlfile, 'cae_weight_final', w)
		utils.modifyXML(xmlfile, 'num_epochs', int(N))
		net = networks_Toy_SS.Toy_CAE_SS(sess, xmlfilename=xmlfile)
		net.build_model()
		net.train_model()
	
	if caseNo == 4:
		xmlfile='../xmls/Toy_hard.xml'
		# first read the xml file
		tree=ET.parse(xmlfile)
		root = tree.getroot()
		# load the graph first
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.75
		sess = tf.Session(config=config)
		# print("For CAE weight : ", w)
		# utils.modifyXML(xmlfile, 'cae_weight_init', w)
		# utils.modifyXML(xmlfile, 'cae_weight_final', w)
		utils.modifyXML(xmlfile, 'num_epochs', int(N))
		net = networks_Toy_SS.Toy_Hard(sess, xmlfilename=xmlfile)
		net.build_model()
		net.train_model()
	# step = 
	# [outLab, outBT] = net.VisualizeModel(step)

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--lambda", type=float, dest="w", default=1, help="reg param")
	parser.add_argument("--num_epochs", type=float, dest="N", default=100, help="num epochs")
	parser.add_argument("--case_number", type=float, dest="caseNo", default=3, help="num epochs")
	args = parser.parse_args()
	temp(**vars(args))
