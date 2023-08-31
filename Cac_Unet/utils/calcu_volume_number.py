'''
融合后的结果-onlyone，与标签计算指标
'''
import glob
import os

from config import settings
import SimpleITK as sitk
import numpy as np


from scipy import ndimage

def compute_calcium_volume_number(spacing, mask_image):

    total_volume = 0

    total_cout = 0

    voxel_volume = np.prod(spacing)


    connectivity = ndimage.generate_binary_structure(3, 3)  # 3,3
    lesion_map, n_lesions = ndimage.label(mask_image, connectivity)

    for lesion in range(1, n_lesions + 1):

        label = np.zeros(mask_image.shape)
        label[lesion_map == lesion] = 1


        lesion_volume = np.count_nonzero(label) * voxel_volume


        pixel_cout = np.sum(label)  # 体素数量
        # calc_volume = pixel_cout * spacing[0] / 3.0 * spacing[1] * spacing[2]  # 钙化体积
        calc_volume = pixel_cout  * spacing[1] * spacing[2]  # 钙化体积

        total_volume += calc_volume
        total_cout += pixel_cout

    return  total_volume, total_cout


def get_tp_fp_tn_fn(spacing,true, pred):


    # 融合，输出指标时用
    true = np.copy(true)
    pred = np.copy(pred)

    #二值化，大于0的全为1
    true[true > 0] = 1
    pred[pred > 0] = 1

    TP = true * pred
    TP_num = np.sum(TP)
    TP_volume,_ =compute_calcium_volume_number(spacing,TP)

    TN =(1-pred) * (1-true)
    TN_num = np.sum(TN)
    TN_volume,_=compute_calcium_volume_number(spacing,TN)


    # FP = pred - TP
    # FN = true - TP

    FN = pred - TP
    FN[FN < 0] = 1
    FN_num= np.sum(FN)
    FN_volume,_=compute_calcium_volume_number(spacing,FN)


    FP = true - TP
    FP[FP < 0] = 1
    FP_num = np.sum(FP)
    FP_volume,_=compute_calcium_volume_number(spacing,FP)


    return TP_volume,TP_num,TN_volume,TN_num,FN_volume,FN_num,FP_volume,FP_num

def read_image(filename, only_data=False):
    """ Reads an mhd file and returns it as a numpy array -> order is (z, y, x) !!! """
    image = sitk.ReadImage(filename)
    data = sitk.GetArrayFromImage(image)

    if only_data:
        return data

    spacing = tuple(reversed(image.GetSpacing()))
    origin = tuple(reversed(image.GetOrigin()))
    return data, spacing, origin


#按单个病人扫描计算-scan，取平均
def patient_metrics(predict, true):
    # print()

    predict_itkimage = sitk.ReadImage(predict)
    predict_np = sitk.GetArrayFromImage(predict_itkimage)
    spacing = tuple(reversed(predict_itkimage.GetSpacing()))

    true_itkimage = sitk.ReadImage(true)
    true_np = sitk.GetArrayFromImage(true_itkimage)

    TP_volume,TP_num,TN_volume,TN_num,FN_volume,FN_num,FP_volume,FP_num = get_tp_fp_tn_fn(spacing,predict_np, true_np)

    return  TP_volume,TP_num,TN_volume,TN_num,FN_volume,FN_num,FP_volume,FP_num



if __name__ == '__main__':

    true_labels = settings.PROCESS_CTALABLELS        #GT
    predict_labels = settings.ONLY_ONE             #predict


    if settings.CHOOSE == 'axis':
        scan = 2677
    elif settings.CHOOSE == 'coronal':
        scan = 2677
        # scan = 2638
    elif settings.CHOOSE == 'sagittal':
        scan = 2677
        # scan = 3169

    predict_files = glob.glob(predict_labels + '/*.nii.gz')

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    TP_volume, TP_num =0,0
    TN_volume, TN_num =0,0
    FN_volume, FN_num =0,0
    FP_volume, FP_num =0,0

    a = len(predict_files)

    for predict_file in predict_files:

        name =os.path.basename(predict_file)
        print(name
              )
        true_file = true_labels +name

        tp_volume,tp_num,tn_volume,tn_num,fn_volume,fn_num,fp_volume,fp_num = patient_metrics(predict_file, true_file)  # 按病人计算，取平均

        TP_volume +=  tp_volume
        TP_num +=  tp_num

        TN_volume += tn_volume
        TN_num += tn_num

        FN_volume += fn_volume
        FN_num += fn_num

        FP_volume += fp_volume
        FP_num += fp_num


    Recall_Volume = TP_volume / (TP_volume + FN_volume)  # 敏感性、召回率、查全率TPR
    precision_Volume = TP_volume / (TP_volume + FP_volume)
    F1_Volume = (2 * precision_Volume * Recall_Volume) / (precision_Volume + Recall_Volume)

    Recall_Num = TP_num / (TP_num + FN_num)  # 敏感性、召回率、查全率TPR
    precision_Num = TP_num / (TP_num + FP_num)
    F1_Num = (2 * precision_Num * Recall_Num) / (precision_Num + Recall_Num)


    # print("tp:{:.4f}".format(TP))
    # print("tn:{:.4f}".format(TN))
    # print("fp:{:.4f}".format(FP))
    # print("fn:{:.4f}".format(FN))

    print("Recall_Volume:{:.4f}".format(Recall_Volume))
    print("PPV_Volume:{:.4f}".format(precision_Volume))
    print("F1_Volume:{:.4f}".format(F1_Volume))

    print("-"*20)
    print("Recall_Num:{:.4f}".format(Recall_Num))
    print("PPV_Num:{:.4f}".format(precision_Num))
    print("F1_Num:{:.4f}".format(F1_Num))

