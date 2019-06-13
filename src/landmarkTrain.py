import numpy as np
import tensorflow as tf
import csv
import xml.etree.ElementTree as ET
import sys


sys.path.append('../ext')
import utils
import networks_landmarks
from utils import show_all_variables
from utils import check_folder
 
xmlfile='../xmls/landmark_drop.xml'
# first read the xml file

	# load the graph first
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.75
sess = tf.Session(config=config)
net = networks_landmarks.Landmark_Dropout_Classification(sess, xmlfilename=xmlfile)
# build the graph
net.build_model()
show_all_variables()
net.train_model()
net.test_model()
