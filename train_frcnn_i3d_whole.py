from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
from keras.utils import plot_model
import os
import cv2
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras.layers import Lambda
from i3d_inception import Inception_Inflated3d
import collections
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import tensorflow as tf
# import keras.backend.tensorflow_backend as ktf

sys.setrecursionlimit(40000)
sys.path.append('/home/subha/hoi_vid/keras-kinetics-i3d')
# def get_session(gpu_fraction=0.333):
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
#                                 allow_growth=True)
#     return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# ktf.set_session(get_session())


old_stdout = sys.stdout

log_file = open("message.log","w")


from i3d_inception import Inception_Inflated3d
# from i3d_inception import Inception_Inflated3d
# from tensorflow.python import keras
from keras.utils import plot_model
import os
import pdb
from keras.layers import Input
import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)
parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
				default="pascal_voc")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=4)
parser.add_option("-s", "--start_idx", type="int", dest="start_idx", help="Number of RoIs to process at once.", default=0)

parser.add_option("-m", "--output_weight_path",dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("-d", "--dataset", dest="dataset", help="Number of RoIs to process at once.", default='AVA')
parser.add_option("-e","--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=2000)

parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--aug", dest="aug", type = int,help="Base network to use. Supports vgg or resnet50.", default=0)
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
				  action="store_true", default=False)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")
parser.add_option("--j", dest="job", help="If the job output should be saved")

(options, args) = parser.parse_args()

if not options.train_path:   # if filename is not given
	parser.error('Error: path to training data must be specified. Pass --path to command line')

if options.parser == 'pascal_voc':
	from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
	from keras_frcnn.simple_parser import get_data
else:
	raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)
C.dataset = options.dataset
C.augment = options.aug
output_weight_path = os.path.join(options.output_weight_path,'model.hdf5')
C.model_path = output_weight_path
C.num_rois = int(options.num_rois)

if options.network == 'vgg':
	C.network = 'vgg'
	from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
	from keras_frcnn import resnet as nn
	C.network = 'resnet50'
else:
	print('Not a valid model')
	raise ValueError


# check if weight path was passed via command line
if options.input_weight_path:
	C.base_net_weights = options.input_weight_path
else:
	# set the path to weights based on backend and model
	C.base_net_weights = nn.get_weight_path()

all_imgs, classes_count, class_mapping = get_data(options.train_path,options.start_idx)

job = options.job
print(len(classes_count)), len(class_mapping)
if job:
	sys.stdout = log_file


# if 'bg' not in classes_count:
# 	classes_count['bg'] = 0
# 	class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename.split('.')[0]+'_'+C.dataset+'.pickle'

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))


data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.image_dim_ordering(), mode='val')

roi_input = Input(shape=(None, 4),name = 'roi_input')
vid_input = Input(shape =(None, None, None, 3),name = 'vid_input')
img_input = Input(shape=(None, None, 3), name = 'img_input')
vid_input_shape = (64, 400,320, 3)
rgb_model = Inception_Inflated3d(
				include_top=False,
				weights='rgb_kinetics_only',
				input_shape=vid_input_shape,
				classes=classes_count)
roi_input = Input(shape=(None, 4),name = 'roi_input')
shared_layers_image = nn.nn_base(img_input, trainable=True)
shared_layers_orig = rgb_model(vid_input)
def slice_tensor(shared_layers):

	feature_shape = shared_layers.shape.as_list()
	shared_layers = shared_layers[:,8,:,:,:]
	return shared_layers

def get_action_dic():

	action_csv = '/work/newriver/subha/AVA_dataset/ava-dataset-tool/ava_action_list_v2.0.csv'
	ac_dic = {}
	f = open(action_csv,'r')
	actions = f.read().splitlines()
	for action in actions[1:]:
		tags = action.split(',')
		tags = tags[:-1]
		ac_id = int(tags[0])
		ac = ''.join(tags[1:])
		if '"' in ac:
			ac =ac.replace('"','')
		# if ',' in ac:
		# 	ac = ''.join(ac.split(','))

		ac_dic[ac_id] = ac
		if ac_id == 1:
			print ac
	return ac_dic

ac_id = get_action_dic()
shared_layers = Lambda(slice_tensor)(shared_layers_orig)
print len(class_mapping)
num_classes = len(class_mapping)
# if C.dataset == 'AVA':
classifier = nn.classifier_i3d_concat(shared_layers_orig, 1, nb_classes=num_classes, trainable=True)

model_classifier = Model([vid_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([vid_input], classifier)
plot_model(model_all, to_file='model_all_i3d_whole.png', show_shapes = True)
log_folder = os.path.join(options.output_weight_path,'logs/')
if not os.path.isdir(log_folder):
	os.makedirs(log_folder)
tensorboard = TensorBoard(log_dir=log_folder)
tensorboard.set_model(model_classifier)
train_names = ['train_loss', 'train_mae']
def write_log(callback, names, logs, batch_no):
	for name, value in zip(names, logs):
		summary = tf.Summary()
		summary_value = summary.value.add()
		summary_value.simple_value = value
		summary_value.tag = name
		callback.writer.add_summary(summary, batch_no)
		callback.writer.flush()

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)

model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_multi_label])

model_all.compile(optimizer='sgd', loss='mae')

epoch_length = 1000
epoch_length_val =100
num_epochs = int(options.num_epochs)
iter_num = 0
iter_num_tensorboard = 0
total_cur_loss = []
total_cur_loss_val = []
iter_num_val_tensorboard = 0
losses = np.zeros((epoch_length, 1))
losses_val = np.zeros((epoch_length_val, 1))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()
###### val #####
rpn_accuracy_rpn_monitor_val = []
rpn_accuracy_for_epoch_val = []

################
best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')
# os.makedirs('check_dataset')
vis = True

for epoch_num in range(num_epochs):

	progbar = generic_utils.Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
	num = 0
	while True:
		try:
			img_data, seq_numpy, x_img = next(data_gen_train)
			Y1 = roi_helpers.calc_label(img_data, C, class_mapping)
			# print X2, Y1
			# x1= (X2[0][0][0])
			# y1 = (X2[0][0][1])
			# x2 = (x1 + X2[0][0][2])
			# y2 = (y1 +X2[0][0][3])
			# x1, y1, x2, y2 = x1*16 , y1*16, x2*16, y2*16
			# # print x1, y1, x2, y2
			# # if x1>320 or x2>320 or y1>400 or y2>400:
			# # 	print "yes"
			# im_temp =cv2.imread(img_data['filepath'])
			# im_temp = cv2.resize(im_temp,(320, 400), interpolation=cv2.INTER_CUBIC)
			# # print im_temp.shape
			# cv2.rectangle(im_temp, (x1,y1),(x2,y2),(0,255,0),3)
			# font = cv2.FONT_HERSHEY_SIMPLEX
			# cl = [i for i, e in enumerate(Y1[0][0]) if e == 1]
			# print cl
			# ind = cl[0]
			# ac =  ac_id[int(class_mapping_inv[ind])]
			#
			# # cv2.putText(im_temp,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)


			# cv2.imwrite(os.path.join('check_dataset',ac+str(num)+'.jpg'),im_temp)

			num+=1

            # img_features =

			loss_class = model_classifier.train_on_batch([seq_numpy], [Y1[:, :, :]])
			losses[iter_num, 0] = loss_class

			iter_num += 1
			write_log(tensorboard,['loss_class'],[loss_class],iter_num_tensorboard)
			iter_num_tensorboard+=1
			progbar.update(iter_num, [('class_loss', np.mean(losses[:iter_num, 0]))])

			if iter_num == epoch_length:
				loss_class_cls = np.mean(losses[:, 0])
				curr_loss =loss_class_cls
				write_log(tensorboard,['total train loss'],[curr_loss],iter_num_tensorboard)
				total_cur_loss.append(curr_loss)
				iter_num = 0
				start_time = time.time()

				#################### Val #########################################################
				iter_num_val = 0

				while True:
						# try:
						img_data, seq_numpy, x_img = next(data_gen_val)
						# print("validation")
						Y1 = roi_helpers.calc_label(img_data, C, class_mapping)
						loss_class = model_classifier.train_on_batch([seq_numpy], [Y1[:, :, :]])
						losses_val[iter_num_val,0] = loss_class
						iter_num_val += 1
						write_log(tensorboard,['loss_class_val'],[loss_class],iter_num_val_tensorboard)
						iter_num_val_tensorboard+=1
						if iter_num_val == epoch_length_val:

							loss_class_cls = np.mean(losses_val[:, 0])
							curr_loss_val = loss_class_cls
							write_log(tensorboard,['total val loss'],[curr_loss_val],iter_num_val_tensorboard)
							total_cur_loss_val.append(curr_loss_val)
							# total_cur_loss.append(curr_loss)
							iter_num_val = 0
							break

				if curr_loss < best_loss:
					if C.verbose:
						print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
					best_loss = curr_loss
					model_all.save_weights(C.model_path)

				break

		except Exception as e:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
			print('Exception: {}'.format(e))
			# print(seq_numpy.shape)
			continue
		sys.stdout = old_stdout

plt.plot(total_cur_loss)
plt.plot(total_cur_loss_val)
plt.legend(['train loss', 'val loss'], loc='upper left')

savefigure = os.path.join(os.path.join(options.output_weight_path,'loss_plot.jpg'))
plt.savefig(savefigure)
print('Training complete, exiting.')
log_file.close()
