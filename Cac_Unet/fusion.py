'''
三个方向3D图像融合为一个3D图像
'''
import os
from glob import glob

from service.Cac_Unet.config import settings
import SimpleITK as sitk

import numpy as np

from scipy import ndimage

# from postprocessing import remove_small_lesions, remove_objects, regiongrow_lesions, read_image


def threedim2one3d(axis_file, coronal_file, sagittaL_file, save_path, filename):

    axis_itkimage = sitk.ReadImage(axis_file)
    axis_ctvalue = sitk.GetArrayFromImage(axis_itkimage)  # 这里一定要注意，得到的是[z,y,x]格式,numpy

    coronal_itkimage = sitk.ReadImage(coronal_file)
    coronal_ctvalue = sitk.GetArrayFromImage(coronal_itkimage)  # 这里一定要注意，得到的是[z,y,x]格式,numpy

    sagittaL_itkimage = sitk.ReadImage(sagittaL_file)
    sagittaL_ctvalue = sitk.GetArrayFromImage(sagittaL_itkimage)  # 这里一定要注意，得到的是[z,y,x]格式,numpy


    #5.
    temp = coronal_ctvalue * sagittaL_ctvalue
    temp += axis_ctvalue


    temp[temp > 0] = 1


    temp_img = sitk.GetImageFromArray(temp)
    sitk.WriteImage(temp_img, os.path.join(save_path, filename))

def fusion():

    axis_path = settings.AXIS_PREDICT_FUSION_SAVE
    coronal_path = settings.CORONAL_PREDICT_FUSION_SAVE
    sagittal_path = settings.SAGITTAL_PREDICT_FUSION_SAVE

    save_path = settings.ONLY_ONE

    axis_files = glob(axis_path + '/*.nii.gz')

    for i, axis_file in enumerate(axis_files):

        axis_filename = os.path.basename(axis_file)
        # print(axis_filename)

        coronal_file = coronal_path + axis_filename
        sagittaL_file = sagittal_path + axis_filename

        threedim2one3d(axis_file, coronal_file, sagittaL_file,  save_path, axis_filename)

    # print("Finish!!!")

if __name__ == '__main__':

    fusion()