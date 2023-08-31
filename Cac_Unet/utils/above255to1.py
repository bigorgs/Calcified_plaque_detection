'''
更改像素值，将像素值大于255的区域设置为1
'''

from PIL import Image
import numpy as np
import os
from config import settings


if __name__ == '__main__':


    if settings.CHOOSE == 'axis':
        work_dir = settings.AXIS_PREDICT_SAVE_PATH
        target_path= settings.AXIS_PREDICT_255
    elif settings.CHOOSE == 'coronal':
        work_dir = settings.CORONAL_PREDICT_SAVE_PATH
        target_path= settings.CORONAL_PREDICT_255
    elif settings.CHOOSE == 'sagittal':
        work_dir = settings.SAGITTAL_PREDICT_SAVE_PATH
        target_path= settings.SAGITTAL_PREDICT_255


    file_names = os.listdir(work_dir)
    for file_name in file_names:
        # print(file_name) # ISIC_0000000_Segmentation.png
        file_path = os.path.join(work_dir, file_name)

        name =os.path.basename(file_name)
        print(name)

        image = Image.open(file_path)
        img = np.array(image)
        img[img == 255] = 1

        # 重新保存
        image = Image.fromarray(img, 'L')
        new_name = file_name[:-4]


        image.save(target_path + new_name +'.png')
