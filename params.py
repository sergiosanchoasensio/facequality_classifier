import time
import datetime
import os
import numpy as np
import pandas as pd
import ast
from sklearn.utils import shuffle


BATCH_SIZE = 16
MAX_STEPS = int(1e+8)
L2_WEIGHT_DECAY = 1e-1
L2_WEIGHT_DECAY_SOFTMAX = 1e-1
TOGGLE_DROPOUT = False
LAMBDAfrn = 1e-3
LAMBDAfrn /= 2 
INITIAL_LEARNING_RATE = 1e-4
DECAY_STEPS = 15000
LEARNING_RATE_DECAY_FACTOR = 0.9
LAMBDAortho = 0.0

AUGMENT = True
PR_RANDOM_CROP = 15  # probability to activate the augmentation (i.e. 15%)
PR_LOW_RES = 15
PR_RANDOM_PATCH = 15

USE_TEMPORAL_INFO = False
ADAM_B1 = 0.9

DEVICE = {}
DEVICE['cuda_id'] = '1'  # for Thomas' machine, 0 = TITAN, 1 = GTX750
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

LABELS = ['frontal', 'profile']
LABELS_BLUR = ['not blurry', 'blurry']

eval_size = 500
DATA_LIST = MAIN_PATH + 'XGBoost_pred_on_vggface2_test.csv'
data = pd.read_csv(DATA_LIST)
data = shuffle(data, random_state=0)
tr_files = list(data['image_path'][data['hand_labeled'] == False])
val_files = list(data['image_path'][data['hand_labeled'] == True])[:eval_size]  # sub sample validation
tr_labels = data['orientation'][data['hand_labeled'] == False]
tr_labels = [ast.literal_eval(t) for t in tr_labels]
tr_labels = np.array([[int(t[x]) for x in t] for t in tr_labels]).astype(np.float32)
val_labels = data['orientation'][data['hand_labeled'] == True]
val_labels = [ast.literal_eval(v) for v in val_labels]
val_labels = np.array([[int(v[x]) for x in v] for v in val_labels]).astype(np.float32)[:eval_size]
ts_files = tr_files[:eval_size]
ts_labels = tr_labels[:eval_size]
if len(LABELS) < 3:  # simplify labelling
    tr_labels_ = []
    for t in tr_labels:
        if t[0] == 1:
            tr_labels_ += [[1.0, 0.0]]
        else:
            tr_labels_ += [[0.0, 1.0]]
    tr_labels = tr_labels_
    val_labels_ = []
    for t in val_labels:
        if t[0] == 1:
            val_labels_ += [[1.0, 0.0]]
        else:
            val_labels_ += [[0.0, 1.0]]
    val_labels = val_labels_
    ts_labels_ = []
    for t in ts_labels:
        if t[0] == 1:
            ts_labels_ += [[1.0, 0.0]]
        else:
            ts_labels_ += [[0.0, 1.0]]
    ts_labels = ts_labels_

N_BOX_TR = len(tr_files)
N_BOX_VAL = len(val_files)
N_BOX_TS = len(ts_files)

NEW_VARIABLES = []

TRAINABLE_VARIABLES = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'fc8', 'outSoftmax']

EVA_FREQ = 1000

LAMBDA_ORIENTATION = 0.5  # how much importance we give to the orientation in the loss
LAMBDA_BLUR = 0.5
