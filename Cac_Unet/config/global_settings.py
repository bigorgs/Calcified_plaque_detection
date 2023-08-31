
import os
from datetime import datetime


EPOCH = 30
MILESTONES = [10, 20, 30]         #start to save best performance model after learning rate decay to 0.01

NUM_WORKERS=0

CHECKPOINT_PATH = '../service/Cac_Unet/checkpoint/'

#coronal 、 sagittal 、 axis
CHOOSE = 'sagittal'

#weight
PREDICT_MODEL = CHECKPOINT_PATH + CHOOSE +'/best_model.pth'


IMAGE_FORMAT = 'png'


#train、valid、test、predict path

#axis
AXIS_TRAIN_IMAGE_PATH = '../service/Cac_Unet/data/Calcification/axis/train/'
AXIS_VALID_IMAGE_PATH = '../service/Cac_Unet/data/Calcification/axis/valid/'
AXIS_TEST_IMAGE_PATH = '../service/Cac_Unet/data/slicers/axis/'


#coronal
CORONAL_TRAIN_IMAGE_PATH = '../service/Cac_Unet/data/Calcification/coronal/train/'
CORONAL_VALID_IMAGE_PATH = '../service/Cac_Unet/data/Calcification/coronal/valid/'
CORONAL_TEST_IMAGE_PATH = '../service/Cac_Unet/data/slicers/coronal/'

#sagittal
SAGITTAL_TRAIN_IMAGE_PATH = '../service/Cac_Unet/data/Calcification/sagittal/train/'
SAGITTAL_VALID_IMAGE_PATH = '../service/Cac_Unet/data/Calcification/sagittal/valid/'
SAGITTAL_TEST_IMAGE_PATH = '../service/Cac_Unet/data/slicers/sagittal/'



#pred to 255
#model_predict_save
#axis
AXIS_PREDICT_SAVE_PATH = '../service/Cac_Unet/data/output/2D/axis/'

#coronal
CORONAL_PREDICT_SAVE_PATH = '../service/Cac_Unet/data/output/2D/coronal/'

#sagittal
SAGITTAL_PREDICT_SAVE_PATH = '../service/Cac_Unet/data/output/2D/sagittal/'




#predict_result_fusion_save
#axis
AXIS_PREDICT_FUSION_SAVE = '../service/Cac_Unet/data/output/2Dto3D/axis/'

#coronal
CORONAL_PREDICT_FUSION_SAVE = '../service/Cac_Unet/data/output/2Dto3D/coronal/'

#sagittal
SAGITTAL_PREDICT_FUSION_SAVE = '../service/Cac_Unet/data/output/2Dto3D/sagittal/'

#3d to  only one 3d save path
ONLY_ONE= '../service/Cac_Unet/data/output/onlyone/'


#Visual
VISUAL= '../service/Cac_Unet/data/output/visual/'


#Visual_label
VISUAL_AXIS= '../service/Cac_Unet/data/output/temp_img/axis/'
VISUAL_CORONAL= '../service/Cac_Unet/data/output/temp_img/coronal/'
VISUAL_SAGITTAL= '../service/Cac_Unet/data/output/temp_img/sagittal/'








