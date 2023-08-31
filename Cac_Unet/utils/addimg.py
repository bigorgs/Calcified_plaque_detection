
from PIL import Image
import numpy as np
import os

if __name__ == '__main__':
    img_dir = "E:\\bigorgs\\Coronary\\2.5D\\CalcificationSeg\\onlyCac\\1_pk\\caijian\\sagittal\\valid\\images/"  # 图像所处文件夹

    lable_dir = "E:\\bigorgs\\Coronary\\2.5D\\CalcificationSeg\\onlyCac\\1_pk\\caijian\\sagittal\\valid\\1/"  # 标签所处文件夹

    target_path = "E:\\bigorgs\\Coronary\\2.5D\\CalcificationSeg\\onlyCac\\1_pk\\caijian\\sagittal\\valid\\2/"

    file_names = os.listdir(img_dir)

    for file_name in file_names:
        # print(file_name)

        file_path = os.path.join(img_dir, file_name)
        name =os.path.basename(file_name)
        print(name)
        image = Image.open(file_path)
        img = np.array(image)

        lable_path =os.path.join(lable_dir, file_name)
        lable = Image.open(lable_path)
        lab = np.array(lable)

        # addimage = img + lab
        addimage = img * lab

        # 重新保存
        addimage = Image.fromarray(addimage, 'L')
        new_name = file_name[:-4]

        # new_name = new_name.strip("_Segmentation")  # 文件名处理成和图像一样的名字


        addimage.save(target_path + new_name +'.png')
