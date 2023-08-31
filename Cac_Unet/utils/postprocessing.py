# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np
from skimage import measure
import SimpleITK as sitk

def read_image(filename, only_data=False):
    """ Reads an mhd file and returns it as a numpy array -> order is (z, y, x) !!! """
    image = sitk.ReadImage(filename)
    data = sitk.GetArrayFromImage(image)

    if only_data:
        return data

    spacing = tuple(reversed(image.GetSpacing()))
    origin = tuple(reversed(image.GetOrigin()))
    return data, spacing, origin

#区域增长
def regiongrow_lesions(mask, img, spacing, max_vol,th):
    mask = mask.astype(int)
    # img_bin = (img > 129).astype(int)
    img_bin = (img > th).astype(int)

    #获取连通区域
    img_labels = measure.label(img_bin)
    ovl_labels, num_lesions = measure.label(mask, return_num=True)

    #计算最大连通区域阈值
    maxvol = np.ceil(max_vol / np.product(spacing)).astype(int)

    for label in range(1, num_lesions + 1):

        location = np.where(ovl_labels == label)

        label_img = []
        for n in range(len(location[0])):
            label_img.append(img_labels[location[0][n], location[1][n], location[2][n]])
        label_img = np.array(label_img)
        label_img = label_img[np.array(np.nonzero(label_img))]
        if len(label_img[0]) > 0:
            label_img = np.argmax(np.bincount(np.asarray(label_img[0])))
        else:
            continue

        location_lesion = np.array(np.where(img_labels == label_img))

        label_mask = []
        for n in range(len(location_lesion[0])):
            label_mask.append(mask[location_lesion[0][n], location_lesion[1][n], location_lesion[2][n]])
        label_mask = np.array(label_mask)
        label_mask = label_mask[np.array(np.nonzero(label_mask))]
        #        print(label_mask)
        label_mask = np.argmax(np.bincount(np.asarray(label_mask[0])))

        # discard if more than 25% of lesion is background
        if float(len(location[0])) / float(len(location_lesion[0])) < 0.25:
            continue

        # discard if lesion grows larger than maximum volume
        if len(location_lesion[0]) > maxvol:
            continue

        for n in range(len(location_lesion[0])):
            mask[location_lesion[0, n], location_lesion[1, n], location_lesion[2, n]] = label_mask
    return mask.astype(int)

#移除小的损伤
def remove_small_lesions(mask, spacing):
    mask_bin = mask > 0
    ovl_labels, num_lesions = measure.label(mask_bin, return_num=True)
    th = (np.ceil(2 / np.product(spacing)).astype(int))
    if th < 2:
        th = 2

    for label in range(1, num_lesions + 1):
        location = np.where(ovl_labels == label)
        if len(location[0]) <= th:
            for n in range(len(location[0])):
                mask[location[0][n], location[1][n], location[2][n]] = 0
    return mask

from scipy import ndimage

#移除较小的候选损伤目标
def remove_objects(binary_mask):
    labelled_mask, num_labels = ndimage.label(binary_mask)  #scipy.ndimage.label() 为每个连接的组件分配不同的标签

    # Let us now remove all the too small regions.
    refined_mask = binary_mask.copy()
    minimum_cc_sum = 2
    #判断区域面积是否小于最小阈值
    for label in range(num_labels):
        if np.sum(refined_mask[labelled_mask == label]) < minimum_cc_sum:
            refined_mask[labelled_mask == label] = 0
    return refined_mask

