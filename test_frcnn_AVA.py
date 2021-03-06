from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras.layers import Lambda
from i3d_inception import Inception_Inflated3d
import collections
from keras_frcnn.utils import *
from pdb import set_trace as bp
from tqdm import tqdm
from keras_frcnn import losses as losses
from keras.optimizers import Adam, SGD, RMSprop
import pandas as pd
from tqdm import tqdm
from pdb import set_trace as bp
from keras.utils.training_utils import multi_gpu_model


'''
 python test_frcnn_AVA.py -m /work/newriver/subha/i3d_models/AVA_exps/multi_label/ -p /work/newriver/subha/AVA_dataset/ava-dataset-tool/preproc/train/keyframes/

'''
sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=4)

parser.add_option("-v", "--val_data", type="str", dest="val_data",
				help="Number of ROIs per iteration. Higher means more memory use.", default='ava_val_subset_80.csv')
parser.add_option("-m", "--model_name", dest="model_name",
				help="Path to model.")
parser.add_option("-o", "--output", dest="output",
				help="csv to save predictions.")
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config_subset_AVA.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')


# config_output_filename = '/home/subha/hoi_vid/keras-kinetics-i3d//keras-frcnn-multi/'
config_output_filename = options.config_filename
with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

if C.network == 'resnet50':
	import keras_frcnn.resnet as nn
elif C.network == 'vgg':
	import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False
resized_width = 320
resized_height = 400
img_path = options.test_path
output_csv_file = os.path.join('evaluation',options.output)
fc = open(output_csv_file,'w+')
def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
	ratio_w = resized_width/width
	ratio_h = resized_height/height
	# if width <= height:
	# 	ratio = img_min_side/width
	# 	new_height = int(ratio * height)
	# 	new_width = int(img_min_side)
	# else:
	# 	ratio = img_min_side/height
	# 	new_width = int(ratio * width)
	# 	new_height = int(img_min_side)
	new_width = resized_width
	new_height = resized_height
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio_w, ratio_h

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio_w, ratio_h= format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio_w, ratio_h

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio_w,ratio_h, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio_w))
	real_y1 = int(round(y1 // ratio_h))
	real_x2 = int(round(x2 // ratio_w))
	real_y2 = int(round(y2 // ratio_h))

	return (real_x1, real_y1, real_x2 ,real_y2)

class_mapping = C.class_mapping
classes_count = class_mapping
if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
# print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)
# print "Num rois originally",C.num_rois
if C.network == 'resnet50':
	num_features = 1024
elif C.network == 'vgg':
	num_features = 512

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	# input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)

shared_layers_input= Input(shape=( None,None,832))
roi_input = Input(shape=(None, 4))
vid_input = Input(shape =(None, None, None, 3))
vid_input_shape = (64, 400,320, 3)
feature_map_input = Input(shape=(None, None,None,832))

rgb_model = Inception_Inflated3d(
				include_top=False,
				weights='rgb_kinetics_only',
				input_shape=vid_input_shape,
				classes=classes_count)
def get_new_img_size(width, height, img_min_side, C):
	img_min_side =448
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	if C.dataset == 'AVA':
		return resized_width, resized_height
	else:
		return 640, 480

def extract_numpy_single_frame(img,C):

	img = (img/255.)*2 - 1
	return img

def get_frame_idx(img_path):
	winSize = 64
	tags = img_path.split(os.path.sep)
	vid_folder = '/'+'/'.join(tags[1:-1])
	frames = os.listdir(vid_folder)
	if 'CAD' in img_path:
		frames = [f for f in frames if f.startswith('RGB')]
		frames.sort(key = lambda x: int(x.split('.')[0].split('_')[1]))
	else:
		frames.sort(key = lambda x: int(x.split('.')[0]))
	frame_index = frames.index(tags[-1])
	fi = get_frames_index(frames,frame_index,winSize)
	seq =[frames[k] if k!=-1 else k for k in fi]
	# print(seq[0],seq[31])
	# print seq
	return seq
optimizer_classifier = Adam(lr=1e-5)
classifier = nn.classifier_i3d_batch(feature_map_input, roi_input, 1, nb_classes=len(classes_count), trainable=True)
# model_classifier_only = Model([feature_map_input, roi_input], classifier)
model_classifier = Model([feature_map_input, roi_input], classifier)
# model_classifier = multi_gpu_model(model_classifier, gpus=2)



model_name = os.path.join(options.model_name,'model.hdf5')
print('Loading weights from {}'.format(model_name))
model_classifier.load_weights(model_name, by_name=True)
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_multi_label])


all_imgs = []
classes = {}
bbox_threshold = 0.7
visualise = True
f_val =  os.path.join('/work/newriver/subha/AVA_dataset/ava-dataset-tool',options.val_data)
df = pd.read_csv(f_val)
# ac_id = get_action_dic()

# for val_vid in val_vids:
final_predictions = []

def get_batch(df, ind):
	rows = df.iloc[ind]
	# print rows
	roi_batch = []
	vid_numpy_batch = []
	for r in range(len(rows)):
		row = rows.iloc[r,:]
		# print row
		# print row[1]
		# bp()
		val_vid = row[0]
		vid_path = os.path.join(img_path,val_vid)
		img_name = str(int(row[1]))+'.jpg'
		k = 0
		k+=1

		if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
			continue
		st = time.time()
		filepath = os.path.join(vid_path,img_name)
		# filepath = '/work/newriver/subha/AVA_dataset/ava-dataset-tool/preproc/train/keyframes/_dBTTYDRdRQ/1589.jpg'
		fr_num = filepath.split(os.path.sep)[-1].split('.')[0]
		img = cv2.imread(filepath)
		# x_img =
		tags = filepath.split(os.path.sep)
		img_folder = '/'+'/'.join(tags[1:-1])
		seq = get_frame_idx(filepath)

		# print filepath, seq
		vid_numpy = []
		x_img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
		for frame in seq:
			if frame!=-1:
				fr_name = os.path.join(img_folder, frame)
				np_name = fr_name.replace('.jpg','.npy')
				np_name = np_name.replace('train/keyframes','numpy_arrays_val')
				# print np_name
				fr_npy = np.load(np_name)
				# fr_img = cv2.imread(fr_name)
				# fr_img =  cv2.resize(fr_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
				# fr_npy = extract_numpy_single_frame(fr_img,C)
				vid_numpy.append(fr_npy)
			else:
				vid_numpy.append(np.zeros((resized_height,resized_width,3)))

		vid_numpy = np.array(vid_numpy)
		# vid_numpy = np.expand_dims(vid_numpy,axis=0)
		# print vid_numpy.shape
		x1,y1,x2,y2 = (float(row[2])*resized_width)/float(16),(float(row[3])*resized_height)/float(16),(float(row[4])*resized_width)/float(16),(float(row[5])*resized_height)/float(16)
		# [x1,y1,x2,y2] = [float(x1)/float(16),float(y1)/float(16), float(x2)/float(16), float(y2)/float(16)]
		w = x2-x1
		h = y2-y1
		roi = np.array([x1,y1,w,h])
		rois = np.expand_dims(roi,axis=0)
		# rois = np.expand_dims(rois,axis=0)
		roi_batch.append(rois)
		vid_numpy_batch.append(vid_numpy)
		# rois = np.expand_dims(rois,axis=0)
	return np.array(roi_batch), np.array(vid_numpy_batch)

indices = range(len(df))
print len(df)
# bp()
bs = 4
for i in tqdm(range(0,len(df),bs)):

	try:
		ind = indices[i:i+bs]
		# row = df.iloc[i,:]
		# val_vid = row[0]
		# vid_path = os.path.join(img_path,val_vid)
		# img_name = str(int(row[1]))+'.jpg'
		# filepath = os.path.join(vid_path,img_name)
		# try:
		rois, vid_numpy = get_batch(df, ind)
		shared_layers_orig = rgb_model.predict(vid_numpy)

		# print rois, shared_layers_orig.shape
		y= model_classifier.predict([shared_layers_orig, rois])
		# print y
		# print y.shape

		# except:
		# 	pass

		# seq_name = filepath.split(os.path.sep)[-2]
		# # bp()
		# # line = [seq_name,str(fr_num).zfill(4),str(float(row[2])),str(float(row[3])),str(float(row[4])),str(float(row[5])),P_cls]
		# # final_predictions.append(line)
		# # for cn in range(P_cls.shape[1]):
		# # 	class_num = cn
		# # 	prob = P_cls[0,cn]
		# # 	class_name = class_mapping[cn]
		# # 	if class_name!='bg':
		# # 		line = seq_name+','+str(fr_num).zfill(4)+','+str(float(row[2]))+','+str(float(row[3]))+','+str(float(row[4]))+','+str(float(row[5]))+','+str(ac_id[class_name])+','+str(prob)
		# # 		# print line
		# # 		f_predicted = open('evaluation/ava_predicted_cheating_subset_latest.csv','a+')
		# # 		f_predicted.write(line+'\n')
		# # 		f_predicted.close()
		# f_predicted = open(output_csv_file,'a+')
		# [f_predicted.write(seq_name+','+str(fr_num).zfill(4)+','+str(float(row[2]))+','+str(float(row[3]))+','+str(float(row[4]))+','+str(float(row[5]))+','+str(class_mapping[cn])+','+str(P_cls[0,cn])+'\n') for cn in range(81) if class_mapping[cn]!='bg']
		# f_predicted.close()

		# if k==1:
		# 	break
	except Exception as e:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print('Exception: {}'.format(e))
		# print(filepath)
		continue
