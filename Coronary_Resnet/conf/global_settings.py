
import os
from datetime import datetime




#model：axis、coronal、sagittal
CHOOSE='coronal'

#directory to save weights file
#axis
AXIS_CHECKPOINT_PATH = '../service/Coronary_Resnet/checkpoint/axis'

#coronal
CORONAL_CHECKPOINT_PATH = '../service/Coronary_Resnet/checkpoint/coronal'

#sagittal
SAGITTAL_CHECKPOINT_PATH = '../service/Coronary_Resnet/checkpoint/sagittal'


#total training epoches
EPOCH = 50
#MILESTONES = [50, 120, 160]        #start to save best performance model after learning rate decay to 0.01
MILESTONES = [10, 20, 30, 40, 50]
# MILESTONES = [5, 15, 30]


#tensorboard log dir
LOG_DIR = 'runs'

IMAGE_FORMAT = 'png'

NUM_WORKERS=0

#model predict output
NUM_CLASSES = 2

#axis
AXIS_TEST_IMAGE_PATH = '../service/Coronary_Resnet/data/3Dto2D/axis'
AXIS_PREDICT_MODEL = '../service/Coronary_Resnet/checkpoint/axis/best.pth'
AXIS_PREDICT_IMAGE_PATH = '../service/Coronary_Resnet/data/3Dto2D/axis/'

#coronal
CORONAL_TEST_IMAGE_PATH = '../service/Coronary_Resnet/data/3Dto2D/coronal'
CORONAL_PREDICT_MODEL = '../service/Coronary_Resnet/checkpoint/coronal/best.pth'
CORONAL_PREDICT_IMAGE_PATH = '../service/Coronary_Resnet/data/3Dto2D/coronal/'

#sagittal
SAGITTAL_TEST_IMAGE_PATH = '../service/Coronary_Resnet/data/3Dto2D/sagittal'
SAGITTAL_PREDICT_MODEL = '../service/Coronary_Resnet/checkpoint/sagittal/best.pth'
SAGITTAL_PREDICT_IMAGE_PATH = '../service/Coronary_Resnet/data/3Dto2D/sagittal/'



#process CTA to only coronary
PROCESS_SAVE_IMAGE='../service/Coronary_Resnet/data/seg'












