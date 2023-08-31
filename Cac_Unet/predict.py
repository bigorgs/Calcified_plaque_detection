import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from service.Cac_Unet.dataset.data_loading import BasicDataset
from service.Cac_Unet.model.Axis.Axis_view import AS_MODEL
from service.Cac_Unet.model.CS.CS_view import CS_MODEL


from service.Cac_Unet.utils.utils import plot_img_and_mask
from service.Cac_Unet.config import settings


def predict_image(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.predict_preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)


    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
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
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_seg.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / (mask.shape[0]-1)).astype(np.uint8))


def predict_img(name , current_num):
    # print("-----start predict--------")

    args = get_args()

    if name == 'axis':
        predict_img = settings.AXIS_TEST_IMAGE_PATH + 'images/'

        predict_pk = settings.AXIS_TEST_IMAGE_PATH + 'PriorKnowledge/'

        save_path = settings.AXIS_PREDICT_SAVE_PATH
        CHECKPOINT_PATH = settings.CHECKPOINT_PATH + name

        net = AS_MODEL(n_channels=1, n_classes=2, bilinear=args.bilinear)

    elif name == 'coronal':
        predict_img = settings.CORONAL_TEST_IMAGE_PATH + 'images/'

        predict_pk = settings.CORONAL_TEST_IMAGE_PATH + 'PriorKnowledge/'

        save_path = settings.CORONAL_PREDICT_SAVE_PATH
        CHECKPOINT_PATH = settings.CHECKPOINT_PATH + name

        net = CS_MODEL(n_channels=1, n_classes=2, bilinear=args.bilinear)

    elif name == 'sagittal':
        predict_img = settings.SAGITTAL_TEST_IMAGE_PATH + 'images/'

        predict_pk = settings.SAGITTAL_TEST_IMAGE_PATH + 'PriorKnowledge/'

        save_path = settings.SAGITTAL_PREDICT_SAVE_PATH
        CHECKPOINT_PATH = settings.CHECKPOINT_PATH + name

        net = CS_MODEL(n_channels=1, n_classes=2, bilinear=args.bilinear)




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    # print(args.model)
    net.to(device=device)


    net.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, 'best_model.pth'), map_location=device))
    logging.info('Model loaded!')

    in_files = glob(predict_img + '/' + current_num +'_*.png')       #读取以当前序号开头的所有png文件

    for i, filename in enumerate(in_files):

        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)

        # 获取文件名
        out_filename = os.path.basename(filename)
        # print(out_filename)


        # 先验知识
        pk_img = predict_pk + out_filename
        pk = Image.open(pk_img)
        pk_np = np.copy(pk)


        probs = predict_image(net=net,
                              full_img=img,
                              scale_factor=args.scale,
                              out_threshold=args.mask_threshold,
                              device=device)

        if not args.no_save:
            result = mask_to_image(probs)
            result_np = np.copy(result)

            # 先验知识融入
            result_pk = result_np * pk_np

            # result.save(save_path + out_filename )        #先验知识处理前

            result_pk = mask_to_image(result_pk)
            result_pk.save(save_path + out_filename)  # 先验知识处理后

            logging.info(f'Mask saved to {save_path + out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, probs)

    # print("\n-----predict completed!!!------")


if __name__ == '__main__':

    name = 'sagittal'
    predict_img(name)
