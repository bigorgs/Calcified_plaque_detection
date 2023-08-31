'''
将3D图像转为2D切片
'''
import os
from glob import glob
import cv2
import numpy as np
from service.Cac_Unet.config import settings
import SimpleITK as sitk



#读取mhd
def read_niigz(niigz_dir):

    itkimage = sitk.ReadImage(niigz_dir)
    ct_value= sitk.GetArrayFromImage(itkimage)  # 这里一定要注意，得到的是[z,y,x]格式

    return ct_value


#设置窗宽、窗位
def set_window(img_data, win_width, win_center):
    img_temp = img_data
    min = (2 * win_center - win_width) / 2.0 + 0.5
    max = (2 * win_center + win_width) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    img_temp = ((img_temp - min) * dFactor).astype(np.int)
    img_temp = np.clip(img_temp, 0, 255)
    return img_temp


def view3Dto2D(file_path):

    axis_save_path = settings.AXIS_TEST_IMAGE_PATH + 'images/'
    cor_save_path = settings.CORONAL_TEST_IMAGE_PATH + 'images/'
    sag_save_path = settings.SAGITTAL_TEST_IMAGE_PATH + 'images/'


    ct_value=read_niigz(file_path)
    # print(ct_value)

    #获取图像三个切面图片
    ct_value_new = set_window(ct_value,750,90)            #窗宽：750；窗位（窗中心）：90


    #获取文件名
    filename = os.path.basename(file_path)
    # print(filename)


    print("---------开始转化------")

    #水平面
    for k in range(ct_value.shape[0]):
        # image
        axis_gray_img = ct_value_new[k, :, :].astype(np.uint8)

        name = axis_save_path + filename[:-7] + '_'+str(k+1)+'.png'
        cv2.imwrite(axis_save_path + filename[:-7] + '_'+str(k+1)+'.png', axis_gray_img )
    print("---------水平面转化完成------")

    #冠状面
    for k in range(ct_value.shape[1]):
        # image
        cor_gray_img = ct_value_new[:, k, :].astype(np.uint8)
        cor_gray_img=np.flipud(cor_gray_img)                 #上下翻转
        cv2.imwrite(cor_save_path+filename[:-7] + '_'+str(k+1)+'.png', cor_gray_img )
    print("---------冠状面转化完成------")

    #矢状面
    for k in range(ct_value.shape[2]):
        #image
        sag_gray_img = ct_value_new[:, :, k].astype(np.uint8)
        sag_gray_img=np.flipud(sag_gray_img)           #上下翻转
        cv2.imwrite(sag_save_path+filename[:-7] + '_'+str(k+1)+'.png', sag_gray_img )
    print("---------矢状面转化完成------")



    print("---------全部转化完成------")

if __name__ == '__main__':

    file_path="../data/source/93.nii.gz"

    view3Dto2D(file_path)

