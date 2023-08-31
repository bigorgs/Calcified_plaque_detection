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


def view3Dto2D2(file_path):
    #获取CT_VALUE

    # file_path=r"./data/slicers/axis/"

    axis_save_path = settings.VISUAL_AXIS
    cor_save_path = settings.VISUAL_CORONAL
    sag_save_path = settings.VISUAL_SAGITTAL


    files = glob(file_path + '/*.nii.gz')

    for file in files:
        ct_value=read_niigz(file)
        # print(ct_value)

        #获取文件名
        filename = os.path.basename(file)
        # print(filename)

        #获取图像三个切面图片
        ct_value_new = set_window(ct_value,750,90)            #窗宽：750；窗位（窗中心）：90

        # print("---------开始转化------")
        # 水平面
        for k in range(ct_value.shape[0]):

            # label,        标签图片需要*255
            axis_gray_img = ct_value[k, :, :].astype(np.uint8)

            axis_gray_img=axis_gray_img *255

            # image-jpg,label-png
            cv2.imwrite(axis_save_path + filename[:-7] + '_' + str(k + 1) + '.png', axis_gray_img)

        # print("---------水平面转化完成------")

        # 冠状面
        for k in range(ct_value.shape[1]):

            # label
            cor_gray_img = ct_value[:, k, :].astype(np.uint8)

            # png
            cor_gray_img = np.flipud(cor_gray_img)  # 上下翻转
            cor_gray_img=cor_gray_img * 255

            cv2.imwrite(cor_save_path + filename[:-7] + '_' + str(k + 1) + '.png', cor_gray_img)

        # print("---------冠状面转化完成------")

        # 矢状面
        for k in range(ct_value.shape[2]):

            # label
            sag_gray_img = ct_value[:, :, k].astype(np.uint8)

            # png
            sag_gray_img = np.flipud(sag_gray_img)  # 上下翻转
            sag_gray_img=sag_gray_img * 255

            cv2.imwrite(sag_save_path + filename[:-7] + '_' + str(k + 1) + '.png', sag_gray_img)
#
#         print("---------矢状面转化完成------")
# print("---------全部转化完成------")



if __name__ == '__main__':

    file_path="../data/output/onlyone/"

    view3Dto2D2(file_path)

