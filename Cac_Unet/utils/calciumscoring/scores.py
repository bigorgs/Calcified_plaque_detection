# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)
import os

import numpy as np

from scipy import ndimage
import SimpleITK as sitk  #读取nii文件

#计算损伤区域的平均密度
# def compute_mean_intensity(image,lesion_mask):
def compute_mean_intensity(image):

    a=image
    # lesion_mask=lesion_mask+0
    # a[lesion_mask ==0]=0

    add=np.sum(a)

    number=np.count_nonzero(a)

    # mean_intensity =np.mean(a)        #no

    if add==0 or number==0:
        return 0

    mean_intensity =add /number

    return mean_intensity



#计算钙化分数
#重写
def compute_calcium_scores(image, spacing, mask_image, labels, min_vol=None, max_vol=None):

    # mask_image = mask[:, :, 0]
    # if np.sum(mask) == 0:
    #     return 0


    agatston_score = 0       #agatston分数
    volume_score = 0        # 体积分数
    mass_score = 0        #质量分数

    total_volume = 0

    total_cout = 0

    connectivity = ndimage.generate_binary_structure(3, 3)  # 3,3
    lesion_map, n_lesions = ndimage.label(mask_image, connectivity)


    for lesion in range(1, n_lesions + 1):

        label = np.zeros(mask_image.shape)
        label[lesion_map == lesion] = 1

        # 计算值为 1 的元素的索引的加权平均值
        idx = np.argwhere(label == 1)
        centroid = idx.mean(axis=0).astype(int)
        z,y,x=centroid

        calc_object = image * label       #钙化图像

        pixel_cout=np.sum(label)        #体素数量
        calc_volume = pixel_cout * spacing[0] / 3.0 * spacing[1] * spacing[2]     #钙化体积

        total_volume +=calc_volume

        total_cout +=pixel_cout

        #1.Agatston分数计算
        object_max = np.max(calc_object)
        agatston = 0
        if 130 <= object_max < 200:
            agatston = calc_volume * 1
        elif 200 <= object_max < 300:
            agatston = calc_volume * 2
        elif 300 <= object_max < 400:
            agatston = calc_volume * 3
        elif object_max >= 400:
            agatston = calc_volume * 4
        agatston_score +=agatston


        #2.体积分数:将识别的体素数量乘以体素体积（以毫米3为单位）
        volume_score += pixel_cout * calc_volume


        #3.质量分数：计算为病变体积与其平均强度的乘积
        image_copy = np.copy(calc_object)
        mean_intensity = compute_mean_intensity(calc_object)
        lesion_mass_score = calc_volume * mean_intensity
        mass_score += lesion_mass_score

    # return agatston_score,volume_score,mass_score
    return agatston_score,total_volume,total_cout,volume_score,mass_score


def compute_calcium_scores2(image, spacing, mask, labels, min_vol=None, max_vol=None):

    binary_mask = np.isin(mask, labels)
    voxel_volume = np.prod(spacing)

    agatston_score = 0       #agatston分数
    calcium_volume = 0      #钙化体积
    volume_score = 0        #体积分数
    mass_score = 0        #质量分数

    # Find individual lesions (in 3D) so that we can discard too small or too large lesions
    connectivity = ndimage.generate_binary_structure(3, 3)       #3,3
    lesion_map, n_lesions = ndimage.label(binary_mask, connectivity)


    for lesion in range(1, n_lesions + 1):
        lesion_mask = lesion_map == lesion            #损伤掩码
        # lesion_mask = lesion_map == 1           #损伤掩码


        #去掉损伤太小或太大的区域
        lesion_volume = np.count_nonzero(lesion_mask) * voxel_volume
        # if min_vol is not None and lesion_volume < min_vol:
        #     continue
        # if max_vol is not None and lesion_volume > max_vol:
        #     continue

        calcium_volume += lesion_volume

        #mass score
        #将标签与图像重合的区域计算平均密度
        mean_intensity =  compute_mean_intensity(image,lesion_mask)         #计算损伤区域的平均密度
        lesion_mass_score = lesion_volume * mean_intensity   #质量分数：计算为病变体积与其平均强度的乘积
        mass_score += lesion_mass_score                     #依次累加



        # Calculate Agatston score for this lesion
        slices = np.unique(np.nonzero(lesion_mask)[0])
        for z in slices:
            fragment_mask = lesion_mask[z, :, :]
            n_pixels = np.count_nonzero(fragment_mask)

            maximum_intensity = np.max(image[z, :, :][fragment_mask])
            #130–199 HU: 1, 200–299 HU: 2, 300–399 HU: 3, ≥400 HU: 4
            if maximum_intensity < 199 and maximum_intensity>130:
                coefficient = 1
            elif maximum_intensity < 299 and maximum_intensity>200:
                coefficient = 2
            elif maximum_intensity < 399 and maximum_intensity>300:
                coefficient = 3
            elif maximum_intensity>=400:
                coefficient = 4
            agatston_score += coefficient * n_pixels

        agatston_score *= spacing[0] / 3.0 * spacing[1] * spacing[2]     #agatston_score

    #volume_score
    n_volume=np.count_nonzero(binary_mask)
    volume_score = calcium_volume * n_volume   #体积分数：将识别的体素数量乘以体素体积（以毫米3为单位）

    return agatston_score, volume_score , mass_score

def seg_volume(filename,image, spacing,origin, mask_image, labels, min_vol=None, max_vol=None):

    image_path="E:\\bigorgs\\cardiovascular\\workspace\\Unet\\Cac_Unet\\output\\patch\\images/"
    label_path="E:\\bigorgs\\cardiovascular\\workspace\\Unet\\Cac_Unet\\output\\patch\\labels/"

    connectivity = ndimage.generate_binary_structure(3, 3)  # 3,3
    lesion_map, n_lesions = ndimage.label(mask_image, connectivity)


    for lesion in range(1, n_lesions + 1):

        label = np.zeros(mask_image.shape)
        label[lesion_map == lesion] = 1

        # 计算值为 1 的元素的索引的加权平均值
        idx = np.argwhere(mask_image == 1)
        centroid = idx.mean(axis=0).astype(int)
        z,y,x=centroid

        image_patch = image[z - 12:z + 12, y - 12:y + 12, x - 12:x + 12]
        label_patch = mask_image[z - 12:z + 12, y - 12:y + 12, x - 12:x + 12]

        outImage = sitk.GetImageFromArray(image_patch)  # numpy 转换成simpleITK
        outImage.SetSpacing(spacing)  # 设置和原来nii.gz文件一样的像素空间
        outImage.SetOrigin(origin)  # 设置和原来nii.gz文件一样的原点位置
        sitk.WriteImage(outImage, os.path.join(image_path, filename[:-7] + '_'+str(lesion)+'.nii.gz'))  # 保存文件

        outLable = sitk.GetImageFromArray(label_patch)  # numpy 转换成simpleITK
        outLable.SetSpacing(spacing)  # 设置和原来nii.gz文件一样的像素空间
        outLable.SetOrigin(origin)  # 设置和原来nii.gz文件一样的原点位置
        sitk.WriteImage(outLable, os.path.join(label_path, filename[:-7] + '_'+str(lesion)+'.nii.gz'))  # 保存文件

    return



def linear_weight_matrix(size):
    j = np.tile(range(size), (size, 1))
    return 1 - np.abs(j.T - j).astype(np.float64) / (size - 1)

#kappa系数
def linearly_weighted_kappa(observed):
    observed = observed.astype('float')
    chance_expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    observed_p = observed / observed.sum()
    chance_expected_p = chance_expected / chance_expected.sum()

    w_m = linear_weight_matrix(observed_p.shape[0])

    p_o = np.multiply(w_m, observed_p).sum()
    p_e = np.multiply(w_m, chance_expected_p).sum()

    kappa_w = (p_o - p_e) / (1 - p_e)
    return kappa_w

#风险类别
def agatston_score_to_risk_category(score):
    categories = [0,100, 400]  # 0-10, 11-100, 101-1000, >1000

    for category, threshold in enumerate(categories):
        if (category+1) == 3:
            if score > threshold:
                return category+2

        if score <= threshold:
            return category + 1

    return len(categories) + 1

#制作混淆矩阵
def make_confusion_matrix(ref_scores, auto_scores):
    max_category = agatston_score_to_risk_category(float('inf'))
    m = np.zeros((max_category, max_category), dtype=int)
    for uid in auto_scores:
        if uid in ref_scores:
            c = agatston_score_to_risk_category(auto_scores[uid]) - 1
            r = agatston_score_to_risk_category(ref_scores[uid]) - 1
            m[r, c] += 1
    return m

#打印混淆矩阵
def print_confusion_matrix(ref_scores, auto_scores):
    m = make_confusion_matrix(ref_scores, auto_scores)

    num_scans = np.sum(m)
    print('{} scan pairs'.format(num_scans))
    print('')

    print('Automatic')
    for r in range(m.shape[0]):
        s = []
        for c in range(m.shape[1]):
            s.append('{}'.format(m[r, c]))
        print('\t'.join(s))
    print('')

    for i in range(len(m)):
        n = np.diagonal(m, i).sum()
        if i > 0:
            n += np.diagonal(m, -i).sum()
        print('{} categories off: {}%'.format(i, n / num_scans * 100))
    print('')

    print('Linearly weighted kappa: {}'.format(linearly_weighted_kappa(m)))
