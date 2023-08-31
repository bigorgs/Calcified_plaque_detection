import numpy as np

def get_tp_fp_tn_fn(true, pred):

    #测试用
    # true = true.cpu().numpy()
    # pred = pred.cpu().numpy()

    # 融合，输出指标时用
    true = np.copy(true)
    pred = np.copy(pred)

    #二值化，大于0的全为1
    true[true > 0] = 1
    pred[pred > 0] = 1

    TP = true * pred
    TN =(1-pred) * (1-true)

    # FP = pred - TP
    # FN = true - TP

    FN = pred - TP
    FP = true - TP


    # a=TP.sum()          #真的正确率
    # b=np.sum(FP)         #误报率
    # c=TN.sum()       #假的正确率
    # d=FN.sum()        #漏报率

    return TP.sum(),FP.sum(),TN.sum(),FN.sum()


def get_F1_score(true, pred):
    """F1"""
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    TP = true * pred
    FP = pred - TP
    FN = true - TP
    precision = TP.sum() / (TP.sum() + FP.sum())
    recall = TP.sum() / (TP.sum() + FN.sum())
    F1 = (2 * precision * recall) / (precision + recall)
    return F1


def get_IoU_score(true, pred):
    """IoU"""
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    TP = true * pred
    FP = pred - TP
    FN = true - TP
    iou = TP.sum() / (TP.sum() + FP.sum() + FN.sum())
    return iou

def get_dice_1(true, pred):
    """Traditional dice."""
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    # 注意这里不是取的并集，而是把两个区域要求和，计算了重叠部分
    denom = true + pred
    return 2.0 * np.sum(inter) / np.sum(denom)

def get_dice_2(true, pred):
    """Ensemble Dice as used in Computational Precision Medicine Challenge."""
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))
    # remove background aka id 0，这里是剔除了背景类
    true_id.remove(0)
    pred_id.remove(0)

    total_markup = 0
    total_intersect = 0
    for t in true_id:
        t_mask = np.array(true == t, np.uint8)
        for p in pred_id:
            p_mask = np.array(pred == p, np.uint8)
            intersect = p_mask * t_mask
            if intersect.sum() > 0:
                total_intersect += intersect.sum()
                total_markup += t_mask.sum() + p_mask.sum()
    return 2 * total_intersect / total_markup