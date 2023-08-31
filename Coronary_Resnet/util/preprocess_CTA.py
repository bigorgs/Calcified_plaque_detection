
'''
根据分类后的冠脉切片边界预处理冠脉CTA
'''

import os
from glob import glob

import SimpleITK as sitk
from service.Coronary_Resnet.conf import settings



# 预处理
def process( ct_path, axis_start,axis_end, coronal_start,coronal_end,sagittal_start,sagittal_end ):

    # 读取图片
    ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    # print("Ori shape:", ct_array.shape)

    # print("Cut out range:[", str(axis_start-1) + ':'  + str(axis_end-1)+','
    #       +str(coronal_start-1) + ':' + str(coronal_end-1)+','
    #       +str(sagittal_start-1) + ':' + str(sagittal_end-1)+']')

    # 截取保留区域
    ct_array = ct_array[axis_start : axis_end, coronal_start : coronal_end , sagittal_start : sagittal_end ]
    # print("Preprocessed shape:", ct_array.shape)

    # 保存为对应的格式（重写ct、seg）
    new_ct = sitk.GetImageFromArray(ct_array)
    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing(ct.GetSpacing())


    return new_ct

def seg_nii(file_path,axis_start,axis_end,coronal_start,coronal_end,sagittal_start,sagittal_end):

    # files = glob(file_path + '/*.nii.gz')
    #
    # for file in files :

    new_ct = process(file_path, axis_start,axis_end,coronal_start,coronal_end,sagittal_start,sagittal_end)

    filename = os.path.basename(file_path)

    #保存
    images_path = os.path.join(settings.PROCESS_SAVE_IMAGE, filename)
    sitk.WriteImage(new_ct, images_path)


if __name__ == '__main__':
    file_path = r"D:\ConarySystem\File\Classify\source/"
    seg_nii(file_path,50, 185,190, 380,110, 370)
    print()
