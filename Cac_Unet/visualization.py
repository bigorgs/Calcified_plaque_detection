'''
可视化标签和预测，与原始图像进行叠加
'''

import os
from glob import glob
from PIL import Image
import numpy as np

import cv2

from service.Cac_Unet.config import settings

def visual():
    file_path = settings.AXIS_TEST_IMAGE_PATH + 'images'
    pred_label_path = settings.VISUAL_AXIS
    save_pred_path = settings.VISUAL

    in_files = glob(file_path + '/*.png')

    for i, file in enumerate(in_files):

        # img = Image.open(file)
        # imgs = np.copy(img)

        # 获取文件名
        name = os.path.basename(file)
        # print(name)

        pred_path = pred_label_path + name
        # true_path=true_label_path+name

        new_name = name[:-4]

        img = cv2.imread(file)  # 读取图片
        pred_label = cv2.imread(pred_path)  # 读取图片
        # true_label = cv2.imread(true_path)  # 读取图片

        # img = cv2.cvtColor(img , cv2.COLOR_GRAY2RGB)        #单通道转三通道
        # img = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)        #三通道转单通道

        shape = img.shape
        height = shape[0]
        width = shape[1]
        flag = False

        pred_img = img.copy()
        true_img = pred_img.copy()

        for h in range(0, height):
            for w in range(0, width):
                if pred_label[h, w, 0] == 255 and pred_label[h, w, 1] == 255 and pred_label[h, w, 2] == 255:
                    pred_img[h, w] = [0,0,255]

                # if true_label[h, w,0] == 255 and true_label[h, w,1] == 255 and true_label[h, w,2] == 255:
                #     true_img[h,w]=[0,255,0]

                    cv2.imwrite(save_pred_path + new_name + ".png", pred_img)

        # cv2.imwrite(save_true_path + new_name +".png" ,true_img)

        # img_pred=img+pred_label
        # img_pred=mask_to_image(img_pred)
        # img_pred.save(save_pred_path + new_name +".png" )
        # cv2.imwrite(save_pred_path + new_name +".png",img_pred)

        # img_true=img+true_label
        # img_true=mask_to_image(img_true)
        # img_true.save(save_true_path + new_name +".png" )
        # cv2.imwrite(save_true_path + new_name +".png" ,img_true)

    return save_pred_path

if __name__ == '__main__':
    visual()
    # print()






