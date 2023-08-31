
#冠脉切片分类

from service.Coronary_Resnet.util.getSlices import view3Dto2D
from service.Coronary_Resnet.predict import model_predict
from service.Coronary_Resnet.util.preprocess_CTA import seg_nii
import os

def Coronary_classify(file_path):

    filename = os.path.basename(file_path)

    #1.视角转换，3Dto2D
    view3Dto2D(file_path)
    print("第一阶段完成")

    #2.三个文件夹同时预测
    view_name = ['axis','coronal','sagittal']
    axis_start,axis_end,coronal_start,coronal_end,sagittal_start,sagittal_end = 0,0,0,0,0,0
    for name in view_name:

        print(name)

        start,end = model_predict(name , filename)
        # print(start,end)

        if name =='axis':
            axis_start,axis_end = start,end
        elif name =='coronal':
            coronal_start,coronal_end = start,end
        elif name == 'sagittal':
            sagittal_start,sagittal_end = start,end

    print(axis_start,axis_end)
    print(coronal_start,coronal_end)
    print(sagittal_start,sagittal_end)
    print("第二阶段完成")

    #3.图像裁剪
    seg_nii(file_path,int(axis_start),int(axis_end), int(coronal_start),int(coronal_end),int(sagittal_start),int(sagittal_end))
    print("第三阶段完成")

    return axis_start,axis_end,coronal_start,coronal_end,sagittal_start,sagittal_end

if __name__ == '__main__':


    file_path = "./data/source/3.nii.gz"

    axis_start,axis_end,coronal_start,coronal_end,sagittal_start,sagittal_end=Coronary_classify(file_path)

    print(axis_start,axis_end,coronal_start,coronal_end,sagittal_start,sagittal_end)




