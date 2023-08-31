import os

from models.myresnet import myresnet34

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from models.resnet import resnet34
from models.resnet import resnet50
from models.densenet import densenet121
import torch
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))#1226x732
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]#112×112    56x56
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
        img=img.astype(np.uint8)  #转成unit8
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
        img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的 56x56x3
        plt.imshow(img)
        print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))

def test(**kwargs):



    # model = densenet121()
    # model = myresnet34()
    model = resnet34()


    # model.classifier = torch.nn.Linear(25600, 2)
    # model.aux_logits = False
    # model = torch.nn.DataParallel(model).cuda()

    # model.classifier = torch.nn.Linear(1024, 2)



    # model.load_state_dict(torch.load('E:/bigorgs/cardiovascular/workspace/Resnet/Coronary_Resnet/checkpoint/axis/myresnet34/myresnet34-49-best.pth'))
    model.load_state_dict(torch.load('E:/bigorgs/cardiovascular/workspace/Resnet/Coronary_Resnet/checkpoint/sagittal/resnet34/resnet34-26-best.pth'))

    if isinstance(model, torch.nn.DataParallel):
        model = model.module



    model.eval()  ## 测试模式，不启用BatchNormalization和Dropout
    dir= 'E:/bigorgs/cardiovascular/workspace/Resnet/Coronary_Resnet/CAM_IMG/256'

    for filename in os.listdir(dir):  # W_67_W_0_8.bmp
        files = filename.split('.')[0]  # W_67_W_0_8
    # 热图生成过程

        savepath = 'E:/bigorgs/cardiovascular/workspace/Resnet/Coronary_Resnet/CAM_IMG/cam'+'/'+files
        # savepath = 'E:/bigorgs/cardiovascular/workspace/Resnet/Coronary_Resnet/CAM_IMG/cam'


        if not os.path.exists(savepath):
            os.mkdir(savepath)
        img = cv2.imread(os.path.join(dir, filename))
        # img = cv2.imread(os.path.join('./CAM_data_256/Second_pic', filename))
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.Normalize(mean=[0.485, 0.456, 0.406],
             #                      std=[0.229, 0.224, 0.225])
             ])
        img = transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():


            # m=model.features


            x_1 = model.conv1(img)
            # draw_features(1, 1, x_1.cpu().numpy(), "{}/stage_1.jpg".format(savepath))

            # # m2=model.outputs
            # # x_base = m.pool0(m.relu0(m.norm0(m.conv0(img))))
            #
            m0 = model.conv2_x(x_1)
            draw_features(1, 1, m0.cpu().numpy(), "{}/stage_2.png".format(savepath))
            # # draw_features(1, 1, m00.cpu().numpy(), "{}/stage_00.jpg".format(savepath))

            m1 = model.conv3_x(m0)
            draw_features(1, 1, m1.cpu().numpy(), "{}/stage_3.png".format(savepath))

            m2 = model.conv4_x(m1)
            draw_features(1, 1, m2.cpu().numpy(), "{}/stage_4.png".format(savepath))

            m3 = model.conv5_x(m2)
            draw_features(1, 1, m3.cpu().numpy(), "{}/stage_5.png".format(savepath))

            # m3 = model.conv5_x(m2)
            # draw_features(1, 1, m0.cpu().numpy(), "{}/stage_0.png".format(savepath))
            #
            # m1 = m.denseblock2(m00)
            # m11 = m.transition2(m1)
            # draw_features(1, 1, m1.cpu().numpy(), "{}/stage_1.png".format(savepath))
            # # draw_features(1, 1, m11.cpu().numpy(), "{}/stage_11.jpg".format(savepath))
            #
            # m2 = m.denseblock3(m11)
            # m22 = m.transition3(m2)
            # draw_features(1, 1, m2.cpu().numpy(), "{}/stage_2.png".format(savepath))
            # # draw_features(1, 1, m22.cpu().numpy(), "{}/stage_22.jpg".format(savepath))
            #
            # m3 = m.denseblock4(m22)
            # m33 = m.norm5(m3)
            # # m333=model.avg_pool2d(model.relu(m33))
            # draw_features(1, 1, m3.cpu().numpy(), "{}/stage_3.png".format(savepath))
            # draw_features(1, 1, m333.cpu().numpy(), "{}/stage_33.jpg".format(savepath))

            #
            # # x_base
            # x_base = model.maxpool(model.relu(model.bn1(model.conv1(img))))
            # # draw_features(1, 1, x_base.cpu().numpy(), "{}/stage_0.jpg".format(savepath))
            #
            #
            # # 1 stage
            # x1 = model.layer1(x_base)
            # draw_features(1, 1, x1.cpu().numpy(), "{}/stage_1.jpg".format(savepath))
            #
            #
            # # 2 stage
            # x2 = model.layer2(x1)
            # draw_features(1, 1, x2.cpu().numpy(), "{}/2.jpg".format(savepath))
            #
            # # 3 stage
            # x3 = model.layer3(x2)
            # draw_features(1, 1, x3.cpu().numpy(), "{}/stage_3.jpg".format(savepath))
            #
            # # 4 stage
            # x4 = model.layer4(x3)
            # draw_features(1, 1, x4.cpu().numpy(), "{}/stage_4.jpg".format(savepath))

            print("done")

        IMAGES_PATH = os.path.join('E:/bigorgs/cardiovascular/workspace/Resnet/Coronary_Resnet/CAM_IMG/cam', files)  # 图片集地址


        IMAGES_FORMAT = ['.png']  # 图片格式
        IMAGE_SIZE = 256  # 每张小图片的大小
        IMAGE_ROW = 2  # 图片间隔，也就是合并成一张图后，一共有几行
        IMAGE_COLUMN = 2  # 图片间隔，也就是合并成一张图后，一共有几列
        result_path='E:/bigorgs/cardiovascular/workspace/Resnet/Coronary_Resnet/CAM_IMG/result'

        if not os.path.exists(result_path):
            os.mkdir(result_path)
        IMAGE_SAVE_PATH = os.path.join(result_path, filename)  # 图片转换后的地址
        # IMAGE_SAVE_PATH =os.path.join('./CAM_data_256/CAM_128' ,filename) # 图片转换后的地址


        # 获取图片集地址下的所有图片名称
        image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
                       os.path.splitext(name)[1] == item]
        image_names.sort()
        #print(image_names)
        print(len(image_names))
        # 简单的对于参数的设定和实际图片集的大小进行数量判断
        if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
            raise ValueError("合成图片的参数和要求的数量不能匹配！")

        # 定义图像拼接函数
        def image_compose():
            to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
            # 循环遍历，把每张图片按顺序粘贴到对应位置上
            for y in range(1, IMAGE_ROW + 1):
                for x in range(1, IMAGE_COLUMN + 1):
                    from_image = Image.open(IMAGES_PATH + '/' + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
            return to_image.save(IMAGE_SAVE_PATH)  # 保存新图

        image_compose()

if __name__ == '__main__':
    # train();
    test();
    # visual();