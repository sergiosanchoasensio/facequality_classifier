import time
import datetime
import os
import numpy as np


BATCH_SIZE = 128
MAX_STEPS = int(1e+8)
L2_WEIGHT_DECAY = 1e-1 #None # 1e-3
L2_WEIGHT_DECAY_SOFTMAX = 1e-1 # None # 1e-3
TOGGLE_DROPOUT = False
LAMBDAfrn = 0.0  # 1e-3
LAMBDAfrn /= 2 
INITIAL_LEARNING_RATE = 1e-4
DECAY_STEPS = 15000
LEARNING_RATE_DECAY_FACTOR = 0.9
LAMBDAortho = 0.0

AUGMENT = False
DATA_AUG_P_gamma = 0.005
DATA_AUG_P_blur = 0.0
DATA_AUG_P_flip = 0.5
DATA_AUG_P_light = 0.005
DATA_AUG_P_chroma = 0.005
DATA_AUG_P_lowres = 0.4
DATA_AUG_P_shittybox = 0.3
DATA_AUG_P_rndpatch = 0.1

USE_TEMPORAL_INFO = False
ADAM_B1 = 0.9

DEVICE = {}
DEVICE['cuda_id'] = '0'  # for Thomas' machine, 0 = TITAN, 1 = GTX750
DEVICE['tf_id'] = '/gpu:' + DEVICE['cuda_id']
if DEVICE['cuda_id']:
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE['cuda_id']

MAIN_PATH = '/nas/datasets/vggface2_test_subset/orientation/'
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
TRAINING_MODEL_DATA_DIR = MAIN_PATH + 'model_data/' + st + '/'
LOGS_PATH = TRAINING_MODEL_DATA_DIR + 'logs/'
ARCH_PARAMS_DIR = TRAINING_MODEL_DATA_DIR + 'arch_params'

OUT_HEIGHT = 128
OUT_WIDTH = 128

NUM_THREADS_FOR_INPUT = 32

TOWER_NAME = 'tower'

EPSILON = np.finfo(float).eps

MOVING_AVERAGE_DECAY = 0.9999

NUM_EPOCHS_PER_DECAY = 36

LABELS = ['high_quality', 'low_quality']


#N_COL = np.sum(COUNT_COLORS)  # number of bounding boxes with color
#N_SUB = np.sum(COUNT_SUBCATS)
#LAMBDA_COLOR = N_SUB / (N_SUB + N_COL)
#LAMBDA_CAT = 1. - LAMBDA_COLOR

RECORDS_PATH = MAIN_PATH + 'tfrecords/'
TRAIN_LIST = MAIN_PATH + 'checknewtrain' # 44k
VAL_LIST = MAIN_PATH + 'checknewtrain' # 0.5k
TRAIN_SMALL_LIST = MAIN_PATH + 'checknewtrain' # 0.5k



i = 0
for i, l in enumerate(open(TRAIN_LIST)): pass
N_BOX_TR = i + 1
i = 0
for i, l in enumerate(open(VAL_LIST)): pass
N_BOX_VAL = i + 1
i = 0
for i, l in enumerate(open(TRAIN_SMALL_LIST)): pass
N_BOX_TS = i + 1

NEW_VARIABLES = []

TRAINABLE_VARIABLES = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'fc8', 'outSoftmax']

