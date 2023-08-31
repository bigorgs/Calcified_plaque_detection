'''
将预测裁剪图映射到原图
'''

import numpy as np
from config import settings
import SimpleITK as sitk


#原图大小全0矩阵创建
#原图大小
z,x,y = 249,512,512
original = np.zeros((z,x,y))
print(original.shape)

#裁剪图预测读取
# label_path = settings.ONLY_ONE
num = '105.nii.gz'

ct_path="E:\\bigorgs\\cardiovascular\\workspace\\Unet\\Cac_Unet\\output\\processCTA\\images\\"+ num
ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
# ct_array = sitk.GetArrayFromImage(ct)

label_path = "E:\\bigorgs\\cardiovascular\\workspace\\Unet\\Cac_Unet\\output\\onlyone\\" + num
label = sitk.ReadImage(label_path, sitk.sitkInt8)
label_array = sitk.GetArrayFromImage(label)
print(label_array.shape)







#取交集，相同位置复制
axis_start, coronal_start, sagittal_start = 36, 240, 100
axis_end, coronal_end, sagittal_end = 195, 390, 340

original[axis_start-1:axis_end-1,coronal_start-1:coronal_end-1,sagittal_start-1:sagittal_end-1]=label_array


#保存图像
save_path="E:\\bigorgs\\cardiovascular\\workspace\\Unet\\Cac_Unet\\output\\onlyone\\save\\"+num
# sitk.WriteImage(original, save_path)

original = sitk.GetImageFromArray(original)
original.SetDirection(ct.GetDirection())
original.SetOrigin(ct.GetOrigin())
original.SetSpacing(ct.GetSpacing())

sitk.WriteImage(original, save_path)


# import nibabel as nib
# nib.save(original,save_path)