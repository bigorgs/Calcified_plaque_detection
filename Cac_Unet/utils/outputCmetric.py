'''
融合后的结果-onlyone，与标签计算指标
'''
import glob
import os

from config import settings
import SimpleITK as sitk

from metrics import get_tp_fp_tn_fn



#按单个病人扫描计算-scan，取平均
def patient_metrics(predict, true):
    # print()

    predict_itkimage = sitk.ReadImage(predict)
    predict_np = sitk.GetArrayFromImage(predict_itkimage)

    true_itkimage = sitk.ReadImage(true)
    true_np = sitk.GetArrayFromImage(true_itkimage)

    tp, fp, tn, fn = get_tp_fp_tn_fn(predict_np, true_np)

    return  tp, fp, tn, fn



if __name__ == '__main__':

    true_labels = settings.PROCESS_CTALABLELS        #GT

    predict_labels = settings.ONLY_ONE             #predict
    # predict_labels = settings.PROCESS_CTALABLELS             #predict
    # predict_labels = settings.AXIS_PREDICT_FUSION_SAVE             #predict
    # predict_labels = settings.SAGITTAL_PREDICT_FUSION_SAVE             #predict
    # predict_labels = settings.CORONAL_PREDICT_FUSION_SAVE             #predict

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

    a = len(predict_files)

    for predict_file in predict_files:

        name =os.path.basename(predict_file)
        true_file = true_labels +name

        tp, fp, tn, fn = patient_metrics(predict_file, true_file)  # 按病人计算，取平均

        TP += int(tp)
        TN += int(tn)
        FN += int(fn)
        FP += int(fp)

    # slicers_metrics()           #按切片计算，求和

    Recall = TP / (TP + FN)  # 敏感性、召回率、查全率TPR
    # specificity = TN / (TN + FP)  # 特异性TNR
    # FPR = FP / (FP + TN)  # 误诊率FPR
    # FNR = FN / (TP + FN)  # 漏诊率FNR
    precision = TP / (TP + FP)
    # F1 = (2 * precision * sensitivity) / (precision + sensitivity)
    ACC = (TP + TN) / (TP + FP + TN + FN)



    # avg_fp = FP / a

    print("tp:{:.4f}".format(TP))
    print("tn:{:.4f}".format(TN))
    print("fp:{:.4f}".format(FP))
    print("fn:{:.4f}".format(FN))
    print("Recall:{:.4f}".format(Recall))
    print("precision:{:.4f}".format(precision))
    print("ACC:{:.4f}".format(ACC))
    print("FP/scan:{:.4f}".format(FP/scan))


