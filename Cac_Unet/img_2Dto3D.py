'''
多个图像按照z轴进行叠加，2D变成3D图像
'''
import os
from glob import glob
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import re

from PIL import Image
import numpy as np
from service.Cac_Unet.config import settings


def natural_sort(l):        #排序
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def img2dto3d(choose_name):

    if choose_name == 'axis':
        source_path = settings.AXIS_PREDICT_SAVE_PATH  # 2d源文件
        save_path = settings.AXIS_PREDICT_FUSION_SAVE  # 保存路径
    elif choose_name == 'coronal':
        source_path = settings.CORONAL_PREDICT_SAVE_PATH
        save_path = settings.CORONAL_PREDICT_FUSION_SAVE
    elif choose_name == 'sagittal':
        source_path = settings.SAGITTAL_PREDICT_SAVE_PATH
        save_path = settings.SAGITTAL_PREDICT_FUSION_SAVE


    files = glob(source_path + '/*.png')

    files=natural_sort(files)         #排序

    images = []

    name = ''  # 当前文件名保存是否更迭

    eof = files[-1]
    eof_name = os.path.basename(eof)  # 最后一个文件的名字

    for file in files:

        filename = os.path.basename(file)
        # print(filename)

        current_name = filename.split('_')[0]
        # print(current_name)

        if name == '' :                #初始化
            name = current_name

        if filename == eof_name:                #最后一张图片加上
            image = Image.open(file)  # 用PIL中的Image.open打开图像
            image_arr = np.array(image)

            images.append(image_arr)

        if name != current_name or filename == eof_name:  # 判断文件名是否改变

            data = np.asarray(images)

            # axis change setting
            # none

            if choose_name == 'sagittal':
                data = data.swapaxes(0,1)
                data = data.swapaxes(1,2)
                data=np.flipud(data)     #上下翻转
            elif choose_name == 'coronal':
                data = data.swapaxes(0, 1)
                data = np.flipud(data)  # 上下翻转

            data_img = sitk.GetImageFromArray(data)

            sitk.WriteImage(data_img, os.path.join(save_path, name+'.nii.gz'))

            images = []     #列表清空

            name = current_name  # 更改文件名


        image = Image.open(file)  # 用PIL中的Image.open打开图像
        image_arr = np.array(image)

        images.append(image_arr)




if __name__ == '__main__':

        choose_name = 'axis'
        #
        # if name == 'axis':
        #     source_path = settings.AXIS_PREDICT_SAVE_PATH          #2d源文件
        #     save_path = settings.AXIS_PREDICT_FUSION_SAVE          #保存路径
        # elif name == 'coronal':
        #     source_path = settings.CORONAL_PREDICT_SAVE_PATH
        #     save_path = settings.CORONAL_PREDICT_FUSION_SAVE
        # elif name == 'sagittal':
        #     source_path = settings.SAGITTAL_PREDICT_SAVE_PATH
        #     save_path = settings.SAGITTAL_PREDICT_FUSION_SAVE


        img2dto3d(choose_name)


