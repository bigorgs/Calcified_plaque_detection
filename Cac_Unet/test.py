import argparse
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.data_loading import BasicDataset, CarvanaDataset
from model.Axis.Axis_view import AS_MODEL
from model.CS.CS_view import CS_MODEL

from model.Unet.unet_model import UNet

from utils.metrics import get_tp_fp_tn_fn

from config import settings
from pathlib import Path


def test(net,test_loader):

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    print("-"*20+"start test!!!"+"-"*20)

    net.eval()
    num_val_batches = len(test_loader)
    print(num_val_batches)

    with torch.no_grad():
        # iterate over the validation set
        for batch in test_loader:
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)                #图片
            mask_true = mask_true.to(device=device, dtype=torch.long)           #标签


            # mask_true=mask_true/255
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()      #转换成one-hot

            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()


            tp, fp, tn, fn = get_tp_fp_tn_fn(mask_pred[:, 1:, ...], mask_true[:, 1:, ...])

            TP += int(tp)
            TN += int(tn)
            FN += int(fn)
            FP += int(fp)



    print("-"*20)

    print("TP:{}".format(TP))
    print("TN:{}".format(TN))
    print("FN:{}".format(FN))
    print("FP:{}".format(FP))

    sensitivity = TP / (TP + FN)  # 敏感性、召回率、查全率TPR
    specificity = TN / (TN + FP)  # 特异性TNR
    FPR = FP / (FP + TN)  # 误诊率FPR
    FNR = FN / (TP + FN)  # 漏诊率FNR
    precision = TP / (TP + FP)
    F1 = (2 * precision * sensitivity) / (precision + sensitivity)
    ACC=(TP+TN)/(TP+FP+TN+FN)

    print("sensitivity:{:.4f}".format(sensitivity))
    print("specificity:{:.4f}".format(specificity))
    print("FPR:{:.4f}".format(FPR))
    print("FNR:{:.4f}".format(FNR))
    print("precision:{:.4f}".format(precision))
    print("F1_score:{:.4f}".format(F1))
    print("ACC:{:.4f}".format(ACC))

    print("-"*20)

    print("-" * 20 + "test ending!!!" + "-" * 20)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=12, help='Batch size')
    parser.add_argument('--model', '-m', default=settings.PREDICT_MODEL, metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=False)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')          #0.5,没用
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')


    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()

    if settings.CHOOSE == 'axis':
        test_img = Path(settings.AXIS_TEST_IMAGE_PATH + 'images/')
        test_mask = Path(settings.AXIS_TEST_IMAGE_PATH + 'labels/')
        h,w = 224,224

    elif settings.CHOOSE == 'coronal':
        test_img = Path(settings.CORONAL_TEST_IMAGE_PATH + 'images/')
        test_mask = Path(settings.CORONAL_TEST_IMAGE_PATH + 'labels/')
        h, w = 336, 336

    elif settings.CHOOSE == 'sagittal':
        test_img = Path(settings.SAGITTAL_TEST_IMAGE_PATH + 'images/')
        test_mask = Path(settings.SAGITTAL_TEST_IMAGE_PATH + 'labels/')
        h, w = 336, 336


    transform = transforms.Compose([
        transforms.Resize((h, w)),
        # transforms.ToTensor(),
    ])

    # 1. Create dataset
    try:
        test_dataset = CarvanaDataset(test_img, test_mask, transform, args.scale)
    except (AssertionError, RuntimeError):
        test_dataset = BasicDataset(test_img, test_mask ,transform, args.scale)

    loader_args = dict(batch_size=args.batch_size, num_workers=settings.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=True, **loader_args)

    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    # net = ResUNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    # net = ResUnet4(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    # net = AS_MODEL(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    # net = CS_MODEL(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    test(net,test_loader)


