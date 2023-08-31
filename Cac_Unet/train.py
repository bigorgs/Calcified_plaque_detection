import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.data_loading import BasicDataset, CarvanaDataset
from model.Axis.Axis_view import AS_MODEL
from model.CS.CS_view import CS_MODEL
from model.ResUnet.Res_Unet import ResUNet
from model.ResUnet.Res_Unet4 import ResUnet4
from model.Unet.unet_model import UNet

from utils.dice_score import dice_loss
from evaluate import evaluate

from  config import settings

import torchvision.transforms as transforms


# train ->valid
def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              # val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False
              ):

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        # transforms.ToTensor(),
    ])


    # 1. Create dataset
    try:
        train_dataset = CarvanaDataset(train_img, train_mask,  transform, img_scale)
        valid_dataset = CarvanaDataset(valid_img, valid_mask,  transform, img_scale)

    except (AssertionError, RuntimeError):
        train_dataset = BasicDataset(train_img, train_mask,  transform, img_scale)
        valid_dataset = BasicDataset(valid_img, valid_mask,  transform, img_scale)


    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)      #10%的训练数据用于验证
    # n_train = len(dataset) - n_val

    n_val = len(valid_dataset)
    n_train = len(train_dataset)

    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=settings.NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(valid_dataset, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-8)    #5e-4
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate,  weight_decay=1e-8)


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)      # goal: maximize Dice score
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=30)      #暂时不管用

    '''
    不改变模型，不降低模型训练精度的前提下
    缩短训练时间、降低存储需求
    '''
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    criterion = nn.CrossEntropyLoss()

    global_step = 0


    best_epoch=0
    best_score=0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        # print("当前学习率：{}".format(optimizer.param_groups[0]['lr']))

        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device)
                true_masks = batch['mask'].to(device)


                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)


                with torch.cuda.amp.autocast(enabled=amp):

                    masks_pred = net(images)

                    loss =  criterion(masks_pred, true_masks) \
                           +  dice_loss(F.softmax(masks_pred, dim=1).float(), F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),multiclass=True)


                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})



                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            # histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            # histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        # if best_score <val_score:
                        if epoch > settings.MILESTONES[0] and best_score <val_score:
                            best_epoch=epoch
                            best_score=val_score
                            torch.save(net.state_dict(), dir_checkpoint +str('best_model_in_epoch{}.pth'.format(epoch)))

                        logging.info('Validation Dice score: {}'.format(val_score))
                        # experiment.log({
                        #     'learning rate': optimizer.param_groups[0]['lr'],
                        #     'validation Dice': val_score,
                        #     'images': wandb.Image(images[0].cpu()),
                        #     'masks': {
                        #         'true': wandb.Image(true_masks[0].float().cpu()),
                        #         'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                        #     },
                        #     'step': global_step,
                        #     'epoch': epoch,
                        #     **histograms
                        # })

        if save_checkpoint and epoch%settings.SAVE_EPOCH == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), dir_checkpoint + str('checkpoint_in_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            print("best_dice_score:{} in {} epoch".format(best_score,best_epoch))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,    #0.01
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')   #0.5
    # parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
    #                     help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')


    return parser.parse_args()


if __name__ == '__main__':

    #choos axis,coronal,sagittal train and valid
    if settings.CHOOSE == 'axis':
        train_path=settings.AXIS_TRAIN_IMAGE_PATH
        valid_path=settings.AXIS_VALID_IMAGE_PATH
        train_img = Path(settings.AXIS_TRAIN_IMAGE_PATH + 'images/')
        valid_img = Path(settings.AXIS_VALID_IMAGE_PATH + 'images/')
        train_mask = Path(settings.AXIS_TRAIN_IMAGE_PATH + 'labels/')
        valid_mask = Path(settings.AXIS_VALID_IMAGE_PATH + 'labels/')
        h,w = 224,224
    elif settings.CHOOSE == 'coronal':
        train_path = settings.CORONAL_TRAIN_IMAGE_PATH
        valid_path = settings.CORONAL_VALID_IMAGE_PATH
        train_img = Path(settings.CORONAL_TRAIN_IMAGE_PATH + 'images/')
        valid_img = Path(settings.CORONAL_VALID_IMAGE_PATH + 'images/')
        train_mask = Path(settings.CORONAL_TRAIN_IMAGE_PATH + 'labels/')
        valid_mask = Path(settings.CORONAL_VALID_IMAGE_PATH + 'labels/')
        h,w = 336,336
    elif settings.CHOOSE == 'sagittal':
        train_path = settings.SAGITTAL_TRAIN_IMAGE_PATH
        valid_path = settings.SAGITTAL_VALID_IMAGE_PATH
        train_img = Path(settings.SAGITTAL_TRAIN_IMAGE_PATH + 'images/')
        valid_img = Path(settings.SAGITTAL_VALID_IMAGE_PATH + 'images/')
        train_mask = Path(settings.SAGITTAL_TRAIN_IMAGE_PATH + 'labels/')
        valid_mask = Path(settings.SAGITTAL_VALID_IMAGE_PATH + 'labels/')
        h,w = 336,336

    print(h,w)

    dir_checkpoint = settings.CHECKPOINT_PATH + settings.CHOOSE +'/'

    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    '''
    Change here to adapt to your data
    n_channels=3 for RGB images
    n_classes is the number of probabilities you want to get per pixel
    '''
    # net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    # net = ResUNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    # net = ResUnet4(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    # net = AS_MODEL(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    net = CS_MODEL(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)


    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    #载入预训练模型
    if args.load:
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  # val_percent=args.val / 100,
                  amp=args.amp,
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise

