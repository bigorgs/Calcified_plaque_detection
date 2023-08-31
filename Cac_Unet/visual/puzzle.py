'''
图像拼接
'''

import os
from glob import glob
from PIL import Image
import numpy as np

import cv2



if __name__ == '__main__':

    save_path = "./contact/"
    true_path="E:\\bigorgs\\cardiovascular\\workspace\\Unet\\Pytorch-UNet-master\\Pytorch-UNet-master\\img_true\\"
    pred_path="E:\\bigorgs\\cardiovascular\\workspace\\Unet\\Pytorch-UNet-master\\Pytorch-UNet-master\\img_pred\\"

    in_files = glob(pred_path + '/*.png')

    for i, file in enumerate(in_files):

        # 获取文件名
        name = os.path.basename(file)
        print(name)

        new_name = name[:-4]

        pred_img = cv2.imread(file)  # 读取图片
        true_img = cv2.imread(true_path+name)  # 读取图片

        contact_img=np.hstack((pred_img,true_img))

        # cv2.imshow('img', contact_img)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        cv2.imwrite(save_path + new_name +".png",contact_img)

