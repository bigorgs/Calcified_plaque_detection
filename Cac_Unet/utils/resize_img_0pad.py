import os
from glob import glob
import numpy as np
from PIL import Image
import cv2


#裁剪图像沿着外边框
def boundaryCrop(img):
    # 获取图片的高，宽
    img_height, img_width = img.shape

    border_top = ( img_height - 512 )//2
    border_bottom = border_top +512

    img_result = img[border_top:border_bottom,:]

    return img_result

# 上下全零填充函数
def boundaryZeroPadding(img):

    # 获取图片的高，宽
    img_height, img_width = img.shape

    border_top=(512 - img_height)// 2
    zeros_array_top = np.zeros((border_top , img_width), dtype=np.uint8)

    if img_height %2 ==0:
        border_bottrom=border_top
    else:
        border_bottrom = border_top + 1

    zeros_array_bottom = np.zeros((border_bottrom  , img_width), dtype=np.uint8)

    # 给图片上下两边添加 0
    img_result = np.concatenate([zeros_array_top,img, zeros_array_bottom], axis=0)

    return img_result






if __name__ == '__main__':
    img_files = "E:\\bigorgs\\cardiovascular\\workspace\\Unet\\Pytorch-UNet-master\\data\\Calcification\\original\\sagittal\\train\\labels\\"
    save_path = "E:\\bigorgs\\cardiovascular\\workspace\\Unet\\Pytorch-UNet-master\\data\\Calcification\\change\\sagittal\\labels\\"

    files = glob(img_files + '/*.png')

    for file in files:
        filename = os.path.basename(file)

        image = Image.open(file)  # 用PIL中的Image.open打开图像
        image_arr = np.array(image)  # 转化成numpy数组

        img_height, img_width = image_arr.shape

        if img_height < 512:
            new_img=boundaryZeroPadding(image_arr)
        elif img_height > 512:
            new_img=boundaryCrop(image_arr)
        else:
            new_img=image_arr

        cv2.imwrite(save_path + filename, new_img)
        print(filename)

    print("finish!!!!")
