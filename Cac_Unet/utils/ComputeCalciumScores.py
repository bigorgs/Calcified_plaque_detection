# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)
import os
from glob import glob

import numpy as np
import pandas as pd

from os import path, makedirs
from time import time
from datetime import datetime, timedelta
from argparse import ArgumentParser

from calciumscoring.datasets import Dataset, read_metadata_file, calcium_labels
from calciumscoring.io import read_image
from calciumscoring.scores import compute_calcium_scores, compute_calcium_scores2, seg_volume

# ----------------------------------------------------------------------------------------------------------------------

# Configuration
from config import settings

config = {
    'model': '2classes',
    'experiment': 'UndilatedDeep65_OTF_FullImage_AllKernels_AllKernelMask',
    'random_seed': 897254,
    'min_vol': 1.5,
    'max_vol': 100000000.0,
}

# Initialization：开始
# overall_start_time = time()

isReference=True
isCCTA=True
isCSCT=True

#冠脉CTA之间：自动-手动
#冠脉CTA与CT之间：手动-手动
#冠脉CTA与CT之间：自动-手动

CCTA_path = settings.CCTA_ROOT
CSCT_path = settings.CSCT_ROOT


image_path = path.join(CCTA_path,"images\\")
# image_path = path.join(CCTA_path,"caijianimages\\")
# image_path = path.join(CSCT_path,"images\\")

images_files = glob(image_path + '/*.nii.gz')

# 计算钙化分数
def calcium_scores(mask):
    # return compute_calcium_scores(image, spacing, mask, label, config['min_vol'], config['max_vol'])
    return seg_volume(filename,image, spacing,origin, mask, label, config['min_vol'], config['max_vol'])

#阈值
threshold = [594.0197,
            479.5947,
            613.1767,
            733.678,
            778.9367,
            432.9527,
            496.8374,
            780.7109,
            339.01,
            715.71,
            692.73,
            638.85,
            852.17,
            405.93,
            ]

# print("序号\t\t\t钙化分数\t总体积\t体素数量\t总体积*体素数量\t体积分数\t质量分数")

#迭代数据集中的所有图像
for i,images_file in enumerate(images_files):

    filename = os.path.basename(images_file)

    mask_path = path.join(CCTA_path,"labels\\")
    # mask_path = path.join(CCTA_path,"caijianlabels\\")
    # mask_path = settings.ONLY_ONE
    # mask_path = path.join(CSCT_path,"labels\\")


    mask_path = mask_path + filename     #标签


    # Calculate calcium scores
    image, spacing, origin = read_image(images_file)
    mask = read_image(mask_path, only_data=True)

    th = threshold[i]

    #设置阈值
    # image[image < 130] = 0
    # image[image < th] = 0
    # mask[image < th] = 0

    a=np.sum(mask)
    # print(a)


    label=1

    # agatston_score,total_volume,total_cout,volume_score,mass_score = calcium_scores(mask)       #计算钙化分数，返回值为agatston分数和病灶体积
    # print("{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}".format(filename,agatston_score,volume_score,mass_score,total_volume,total_cout))

    calcium_scores(mask)

    # agaston_score , total_volume , total_cout= calcium_scores(mask)       #计算钙化分数，返回值为agatston分数和病灶体积
    # print("{}\t{:.1f}\t{:.1f}\t{:.1f}".format(filename,agaston_score,total_volume * total_cout,total_cout))





