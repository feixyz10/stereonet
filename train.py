import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from copy import deepcopy
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

torch.backends.cudnn.benchmark = True

from model import StereoNet
from dataset import StereoDataset, CenterCrop, ToTensor, BottomLeftCrop

parser = argparse.ArgumentParser(description="Train StereoNet.")
parser.add_argument('--checkpoint', default='', type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--dataset', default='kitti', type=str)
parser.add_argument('--batch', default=1, type=int)
parser.add_argument('--dmax', default=192, type=int)
parser.add_argument('--nodebug', action='store_false', default=True)
parser.add_argument('--logid', default=10, type=int)
parser.add_argument('--log_freq', default=100, type=int)
parser.add_argument('--save_freq', default=5000, type=int)
parser.add_argument('--iter', default=2, type=int)
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
dmax = args.dmax
if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)

model = StereoNet(K=3, dmax=dmax, R=3)
if args.checkpoint:
    model.load_state_dict(checkpoint['model'])
model.to(device)
model.train()

if args.dataset == 'kitti':
    img_lft_fns = [x.strip() for x in open('data/kitti_disp_train_left_filenames.txt').readlines()]
    img_rgt_fns = [x.strip() for x in open('data/kitti_disp_train_right_filenames.txt').readlines()]
    disp_fns = [x.strip() for x in open('data/kitti_disp_train_disp_filenames.txt').readlines()]
    image_size = [320, 960]
    trans = transforms.Compose([CenterCrop(image_size), ToTensor()])
elif args.dataset == 'sceneflow':
    img_lft_fns = [x.strip() for x in open('data/sceneflow_train_left_filenames.txt').readlines()]
    img_rgt_fns = [x.strip() for x in open('data/sceneflow_train_right_filenames.txt').readlines()]
    disp_fns = [x.strip() for x in open('data/sceneflow_train_disp_filenames.txt').readlines()]
    image_size = [480, 800]
    trans = transforms.Compose([CenterCrop(image_size), ToTensor()])

dataset = StereoDataset(img_lft_fns, img_rgt_fns, disp_fns, trans)
if args.nodebug:
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=6, pin_memory=True)
else:
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

criterion = nn.SmoothL1Loss()
params_to_update = model.parameters()
optimizer = torch.optim.RMSprop(params_to_update, lr=args.lr)
if args.checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
writer = SummaryWriter('runs/disp_train_%s_%d'%(args.dataset, args.logid))

start_epoch = checkpoint['epoch'] if args.checkpoint else 1
start_iter_num = checkpoint['iteration'] if args.checkpoint else 0

def plot_figure(image, disp, preds):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(2, 3, 1, xticks=[], yticks=[])
    img = ((image[0].cpu().detach().numpy() + 1) / 2 * 255).astype(np.uint8).transpose([1,2,0])
    ax.set_title('left image')
    plt.imshow(img)
    ax = fig.add_subplot(2, 3, 2, xticks=[], yticks=[])
    ax.set_title('groundtruth')
    img = disp[0].cpu().detach().numpy()
    plt.imshow(img)
    for i in range(3, 7):
        ax = fig.add_subplot(2, 3, i, xticks=[], yticks=[])
        img = preds[i-3][0].cpu().detach().numpy()
        ax.set_title('prediction_%d'%(i-3))
        plt.imshow(img)
    fig.tight_layout()
    return fig
    

def train(model, criterion, optimizer, dataloader, writer=None, epochs=1, warm_up=True, lr_scheduler=None, log_freq=1, save_freq=1, start_epoch=1, start_iter_num=0):
    lr = [param_group['lr'] for param_group in optimizer.param_groups]
    if warm_up:
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr[i] / 100
    running_loss = 0.0
    running_loss_stages = None
    running_num = 0
    iter_num = start_iter_num
    for epoch in range(start_epoch, epochs+1):
        print([param_group['lr'] for param_group in optimizer.param_groups])
        pbar = tqdm(total=len(dataloader)) 
        for sample in dataloader:
            iter_num += 1
            running_num += 1
            if warm_up and (iter_num - start_iter_num) % 500 == 499:
                for i, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = lr[i]
                warm_up = False
            img_lft = sample['left'].to(device)
            img_rgt = sample['right'].to(device)
            disp = sample['disparity'].to(device)
            optimizer.zero_grad()
            preds = model(img_lft, img_rgt)
            mask = (disp > 0) * (disp <= dmax)
            losses = [criterion(pred[mask], disp[mask]) for pred in preds]
            loss = sum(losses) / len(losses)
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                if running_loss_stages is not None:
                    running_loss_stages = [x + y.item() for x, y in zip(running_loss_stages, losses)]
                else:
                    running_loss_stages = [y.item() for y in losses]
                running_loss += loss.item()
            
            pbar.update(dataloader.batch_size)
            pbar.set_description('[Epoch %d, Running Loss: %.4f]' % (epoch, running_loss / running_num))

            if iter_num % save_freq == 0:
                checkpoint = {'model': deepcopy(model).to('cpu').state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'iteration': iter_num}
                torch.save(checkpoint, 'checkpoints/stereonet_%d_%d_iter%d.ckpt.tar'%(epoch, iter_num, args.iter))
            if writer is not None and iter_num % log_freq == 0:
                writer.add_scalars('traing_loss', {'all': running_loss / running_num}, iter_num)
                for i, loss_stage in enumerate(running_loss_stages, 1):
                    writer.add_scalars('traing_loss', {'loss_%d'%(i): loss_stage / running_num}, iter_num)
                fig = plot_figure(img_lft, disp, preds)
                writer.add_figure('label vs. pred', fig, iter_num)
                running_loss = 0.0 
                running_num = 0
                running_loss_stages = None          
                
        pbar.close()
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] *= 0.6


warm_up = False if args.checkpoint else True
train(model, criterion, optimizer, dataloader, writer, epochs=args.epochs, warm_up=warm_up, lr_scheduler=None, log_freq=args.log_freq, save_freq=args.save_freq, start_epoch=start_epoch, start_iter_num=start_iter_num)