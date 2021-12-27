import open3d as o3d
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from torchvision import transforms
import cv2
from skimage import io
import matplotlib.pyplot as plt
import argparse
import numpy as np

from model import StereoNet
from dataset import StereoDataset, CenterCrop, ToTensor, BottomLeftCrop

parser = argparse.ArgumentParser(description="Test StereoNet.")
parser.add_argument('--weight', default='checkpoints/stereonet_180000_gpu5.pth', type=str)
parser.add_argument('--checkpoint', default='checkpoints/stereonet_4_180000.ckpt.tar', type=str)
parser.add_argument('--dataset', default='sceneflow', type=str)
parser.add_argument('--dmax', default=192, type=int)
parser.add_argument('--iter', default=2, type=int)
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
dmax = args.dmax

checkpoint = torch.load(args.checkpoint)
# checkpoint = {'model': torch.load(args.weight)}

R = 3
model = StereoNet(K=3, dmax=dmax, R=R)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()
# model.train()
for param in model.parameters():
    param.requires_grad = False

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
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)


P = np.array([
    [1.09331779e+03, 0.00000000e+00, 5.53095516e+02, 0.00000000e+00], 
    [0.00000000e+00, 1.09331779e+03, 2.90187443e+02, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]
    ])
bf = 541.207280531

def point2D_to_point3D(point2d, disp, P, bf):
    t = np.linalg.inv(P[:, :3]).dot(P[:, 3]) # translation vector
    fx, fy, cx, cy = P[0,0], P[1,1], P[0,2], P[1,2]
    Z = bf / disp
    X = (point2d[:,0] - cx) / fx * Z - t[0]
    Y = (point2d[:,1] - cy) / fy * Z - t[1]
    Z = Z - t[2]

    return np.stack([X, Y, Z], 1) 


def pseudo_lidar(disp, P, bf, image=None, offset=[0, 0]):
    h, w = disp.shape
    disp = disp.flatten()
    mask = (disp > 0)
    disp = disp[mask]
    
    u = np.arange(w, dtype=np.float32) + offset[1]
    v = np.arange(h, dtype=np.float32) + offset[0]
    u, v = np.meshgrid(u, v)
    uv = np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)])
    point2d = uv[mask]

    point3d = point2D_to_point3D(point2d, disp, P, bf)
    color = image.reshape(-1, 3)[mask]

    return np.hstack([point3d, color])


def bird_eye_view(cloud, x_range=[-28, 28], y_range=[-1, 3], z_range=[0, 80], resolution=0.1):
    mask = (x_range[0] < cloud[:, 0]) * (cloud[:, 0] < x_range[1]) * (y_range[0] < cloud[:, 1]) * (cloud[:, 1] < y_range[1]) * (z_range[0] < cloud[:, 2]) * (cloud[:, 2] < z_range[1])  
    cloud = cloud[mask]
    
    sort_idx = np.argsort(cloud[:, 1])[::-1]
    cloud = cloud[sort_idx]

    l = int((z_range[1] - z_range[0]) / resolution)
    w = int((x_range[1] - x_range[0]) / resolution)
    
    view = np.zeros([l, w, 3])
    grid_x = ((cloud[:, 0] - x_range[0]) / resolution).astype(np.int32)
    grid_z = ((cloud[:, 2] - z_range[0]) / resolution).astype(np.int32)
    view[grid_z, grid_x] = cloud[:, 3:]

    return view


fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
# ax1 = fig.add_subplot(1, 2, 2)#, xticks=[], yticks=[])
# ax1.set_title('BEV')
# ax2 = fig.add_subplot(2, 2, 1, xticks=[], yticks=[])
# ax2.set_title('left image')
# ax3 = fig.add_subplot(2, 2, 3, xticks=[], yticks=[])
# ax3.set_title('Disparity')

for sample in dataloader:
    img_lft = sample['left'].to(device)
    img_rgt = sample['right'].to(device)
    disp = sample['disparity'].to(device)
    mask = disp > 0
    preds = model(img_lft, img_rgt)

    # left_img = ((img_lft[0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8).transpose([1,2,0])
    # ax2.imshow(left_img)
    # disp = preds[-1][0].cpu().numpy()
    # ax3.imshow(disp)
    # cloud = pseudo_lidar(disp, P, bf, left_img, offset=[8,0])
    # bev = bird_eye_view(cloud).astype(np.uint8)[::-1]
    # ax1.imshow(bev)

    left_img = ((img_lft[0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8).transpose([1,2,0])
    ax = fig.add_subplot(2, 3, 1, xticks=[], yticks=[])
    plt.imshow(left_img)
    ax = fig.add_subplot(2, 3, 2, xticks=[], yticks=[])
    img = disp[0].cpu().numpy()
    plt.imshow(img)
    ax.set_title('Dmax={0:.2f}, Dmin={1:.2f}'.format(disp[mask].max().item(), disp[mask].min().item()))

    for i in range(3, 4+R):
        ax = fig.add_subplot(2, 3, i, xticks=[], yticks=[])
        img = preds[i-3][0].cpu().numpy()
        plt.imshow(img)

    EPE = F.l1_loss(preds[-1][mask], disp[mask]).item()
    num = mask.sum().to(torch.float32).item()
    d3e = (torch.abs(disp[mask] - preds[-1][mask]) > 3).sum().item() / num 
    d1e = (torch.abs(disp[mask] - preds[-1][mask]) > 1).sum().item() / num 
    ax.set_title('Dmax={0:.1f}, Dmin={1:.1f}, EPE={2:.2f}, D3E={3:.2f}%, D1E={4:.2f}%'.format(preds[-1][mask].max(), preds[-1][mask].min(), EPE, d3e * 100, d1e * 100))
    
    plt.draw()
    plt.tight_layout()

    # pcd = pseudo_lidar(preds[-1][0].cpu().numpy(), P, bf, left_img/255)
    # o3d.visualization.draw_geometries([pcd])

    plt.waitforbuttonpress(0)
    # plt.show()