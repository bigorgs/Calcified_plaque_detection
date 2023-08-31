import os
from glob import glob

from service.Cac_Unet.predict import predict_img
from service.Cac_Unet.fusion import fusion
from service.Cac_Unet.img_2Dto3D import img2dto3d
from service.Cac_Unet.utils.getSlices2 import view3Dto2D2
from service.Cac_Unet.visualization import visual
from service.Cac_Unet.utils.getSlices import view3Dto2D
from service.Cac_Unet.config import settings

#冠脉斑块检测
def Plaque_detection(file_path):

    filename = os.path.basename(file_path)
    current_num = filename.split('.')[0]           #获取当前文件序号

    #1.视角转换，3Dto2D
    view3Dto2D(file_path)
    print("第一阶段完成")


    #2.三个视角同时预测
    view_name = ['axis','coronal','sagittal']
    for name in view_name:
        # print(name)

        predict_img(name ,current_num )   # 预测

        img2dto3d(name)   #2D转3D

    print("第二阶段完成")


    #3.多视角融合，2Dto3D
    fusion()
    print("第三阶段完成")

    #4.视角转换
    view3Dto2D2(settings.ONLY_ONE)
    print("第四阶段完成")


    #5.可视化
    save_path = visual()
    print("第五阶段完成")


    #读取文件夹
    files_list = glob(save_path + '*.png')

    return files_list




if __name__ == '__main__':

    file_path = "./data/source/93.nii.gz"

    files_list = Plaque_detection(file_path)




