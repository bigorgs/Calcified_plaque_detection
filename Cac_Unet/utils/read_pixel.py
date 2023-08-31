import os
from glob import glob
from PIL import Image
import numpy as np
from shutil import copyfile


image_path="E:\\bigorgs\\cardiovascular\\workspace\\Unet\\Cac_Unet\\data\\Calcification\\sagittal\\valid\\labels/"

copy_path= "E:\\bigorgs\\Coronary\\2.5D\\priorknowledge\\unnormal\\caijian\\sagittal/"

save_path= "E:\\bigorgs\\cardiovascular\\workspace\\Unet\\Cac_Unet\\data\\Calcification\\sagittal\\valid\\1/"


files = glob(image_path+'/*.png')

cout=0

for file in files:

    print(file)

    img = Image.open(file)
    #img.show()

    # 获取文件名
    filename = os.path.basename(file)
    # 对应得image路径
    image_file = copy_path + filename

    img_array = np.array(img)           #把图像转成数组格式img = np.asarray(image)
    shape = img_array.shape
    height = shape[0]
    width = shape[1]
    flag=False
    for h in range(0,height):
        for w in range (0,width):
            if img_array[h,w] == 255:          #白色
                cout=cout+1
                flag=True
                print("---------------------------------------------")
                print("开始复制！！！！")
                print("cout={}".format(cout))
                print("文件名：{}".format(filename))

                #原始图片复制
                image_target_path= save_path + filename
                copyfile(image_file, image_target_path)

                #标签文件复制
                # label_target_path=target_path_label+filename
                # copyfile(file, label_target_path)
                print("---------------------------------------------")
                break
        if flag:
            flag=False
            break


print("cout={}".format(cout))
print("finish!!!!")

# if __name__ == '__main__':
