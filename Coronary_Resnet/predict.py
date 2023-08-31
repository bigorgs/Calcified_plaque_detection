
import argparse

from service.Coronary_Resnet.utils import get_network

import glob
import os

from service.Coronary_Resnet.conf import settings
import torch

from PIL import Image

import torchvision.transforms as transforms
import re



transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def natural_sort(l):        #排序
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def predict_images(image_file, model):

    image = Image.open(image_file)
    image = image.convert("RGB")
    image = transform_test(image)
    image = image.unsqueeze_(0).to(device)
    with torch.no_grad():
        outputs = model(image)

        outputs = outputs.to(device)
    predict_label = torch.max(outputs, dim=1)[1].data.cpu().numpy()[0]



    #冠脉类别图片序号输出
    # if label == 1  and predict_label == 1:
    if predict_label == 1:
        return True
    else:
        return False



def get_image_label_to_predict(predict_path , filename):

    net.eval()

    current_num = filename.split('.')[0]

    images = glob.glob(os.path.join(predict_path, current_num+ "_*.{}".format(settings.IMAGE_FORMAT)))

    images = natural_sort(images)

    list=[]

    for img in images:

        result = predict_images(img, net)

        if  result:
            name = os.path.basename(img)
            # print("文件名："+name)

            file_name = name.split('.')
            number_split = file_name[0].split('_')[1]       #下划线前的数字
            # print("文件序号：" + number_split)

            list.append(number_split)

    start = list[0]
    end = list[-1]

    # print("第一张图像：{}".format(list[0]))
    # print("最后一张图像：{}".format(list[-1]))


    return start,end


def model_predict(name,filename):

    # if settings.CHOOSE == "axis":
    if name == "axis":

        predict_path = settings.AXIS_PREDICT_IMAGE_PATH
        CHECKPOINT_PATH = settings.AXIS_CHECKPOINT_PATH

    elif name == "coronal":

        predict_path = settings.CORONAL_PREDICT_IMAGE_PATH
        CHECKPOINT_PATH = settings.CORONAL_CHECKPOINT_PATH

    elif name == "sagittal":

        predict_path = settings.SAGITTAL_PREDICT_IMAGE_PATH
        CHECKPOINT_PATH = settings.SAGITTAL_CHECKPOINT_PATH

    args.weights = os.path.join(CHECKPOINT_PATH, args.net, 'best_model.pth')

    net.load_state_dict(torch.load(args.weights))

    start, end = get_image_label_to_predict(predict_path , filename)

    return start,end


#网络配置参数
parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, default='resnet34', help='net type')
parser.add_argument('-weights', type=str, help='the weights file you want to test')
parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
parser.add_argument('-b', type=int, default=12, help='batch size for dataloader')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

net = get_network(args)

# RESNET
# num_fits = net.fc.in_features
# net.fc = nn.Linear(num_fits, settings.NUM_CLASSES)

net.to(device)



if __name__ == '__main__':
    start, end = model_predict('coronal')
    print(start,end)
