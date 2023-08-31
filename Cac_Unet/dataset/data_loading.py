import glob
import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from service.Cac_Unet.config import settings


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str ,data_transforms=None,scale: float = 1.0, mask_suffix: str = ''):

        self.data_transforms = data_transforms

        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(self,pil_img, scale, is_mask):
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)

        pil_img = self.data_transforms(pil_img)

        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def predict_preprocess(pil_img, scale, is_mask):
        # w, h = pil_img.size
        # # newW, newH = int(scale * w), int(scale * h)

        if settings.CHOOSE == 'axis':
            newW, newH = 224,224
        elif settings.CHOOSE == 'coronal':
            newW, newH = 336,336
        elif settings.CHOOSE == 'sagittal':
            newW, newH = 336,336

        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)

        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))


        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])


        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'


        img = self.preprocess(self, img,  self.scale, is_mask=False)
        mask = self.preprocess(self, mask,  self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            # 'image':img ,
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            # 'mask': mask

        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, transform, scale=1):
        # super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
        super().__init__(images_dir, masks_dir, transform, scale,  mask_suffix='')


def default_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class Dataset:

    def __init__(self, image_root_path, data_transforms=None, image_format='png'):
        self.data_transforms = data_transforms
        self.image_root_path = image_root_path
        self.image_format = image_format
        self.images = []
        self.labels = []

        classes_folders = os.listdir(self.image_root_path)
        for cls_folder in classes_folders:
            folder_path = os.path.join(self.image_root_path, cls_folder)
            if os.path.isdir(folder_path):
                images_path = os.path.join(folder_path, "*.{}".format(self.image_format))
                images = glob.glob(images_path)
                self.images.extend(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_file = self.images[item]
        lable_file = self.labels[item]


        # label_name = os.path.basename(os.path.dirname(image_file))
        image = default_loader(image_file)
        lable = default_loader(lable_file)

        if self.data_transforms is not None :
            image = self.data_transforms(image)
            lable = self.data_transforms(lable)


        return image, lable
