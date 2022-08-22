import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import random
import time
import numpy as np

import utils
from dataloaders.data_rgb import get_training_data, get_validation_data
from pdb import set_trace as stx

from tqdm import tqdm
import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
import glob
import time
import scipy.io
from utils import utils_logger
from utils import utils_image as util
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from models.loss_ssim import SSIMLoss
from models.network_mhcnn_color import Net

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
sigma = 'realnoise'
batch_size = 32
patch_sizes = 128
in_channels = 3
hiddens = 128
lr = 0.0001
test_iter = 1000
load_pretrained = True

seed = 100
# logger.info('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_dir = r'E:\image_denoising\aaaa\SIDD_patches\train'

def test(model):
    model.eval()
    avg_psnr = 0
    avg_ssim = 0

    count = 0
    torch.manual_seed(0)
    i_imgs, i_blocks, _, _, _ = all_noisy_imgs.shape
    psnrs = []
    ssims = []
    import utils.utils_image as util
    for i_img in range(i_imgs):
        for i_block in range(i_blocks):
            noise = transforms.ToTensor()(Image.fromarray(all_noisy_imgs[i_img][i_block])).unsqueeze(0)
            noise = noise.cuda()
            noisy_flip = noise.flip(-2)
            noisy_rot = torch.rot90(noise, k=1, dims=[-1,-2])
            # noisy_flip = torch.rot90(noise, k=3, dims=[-1, -2])  # 270度
            # noisy_rot = torch.rot90(noise, k=1, dims=[-1, -2])
            with torch.no_grad():
                pred = model(noise,noisy_flip,noisy_rot)
            pred = pred.detach().float().cpu()
            gt = transforms.ToTensor()((Image.fromarray(all_clean_imgs[i_img][i_block])))
            gt = gt.unsqueeze(0)
            pred = util.tensor2uint(pred)
            gt = util.tensor2uint(gt)
            psnr_t = util.calculate_psnr(pred, gt)
            ssim_t = util.calculate_ssim(pred, gt)
            psnrs.append(psnr_t)
            ssims.append(ssim_t)
            avg_psnr += psnr_t
            avg_ssim += ssim_t
            count += 1
    avg_psnr = avg_psnr / count
    avg_ssim = avg_ssim / count
    return avg_psnr, avg_ssim, psnrs


######### Model ###########
if __name__ == '__main__':
    mhcnn = Net(in_channels,in_channels,hiddens).cuda()
    mhcnn = torch.nn.DataParallel(mhcnn)
    if load_pretrained:
        mhcnn.load_state_dict(torch.load('model_zoo\mhcnn.pth'))

    all_noisy_imgs = scipy.io.loadmat(r'E:\image_denoising\zzz-finished\DRNet\DRNet\DR_new\testsets\ValidationNoisyBlocksSrgb.mat')[
    'ValidationNoisyBlocksSrgb']
    all_clean_imgs = scipy.io.loadmat(r'E:\image_denoising\zzz-finished\DRNet\DRNet\DR_new\testsets\ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb']

    best = 0
    count = 0

    test_val_, avg_ssim, psnr = test(mhcnn)
    print(test_val_, avg_ssim)

