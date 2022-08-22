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
batch_size = 64
patch_sizes = 128
in_channels = 3
hiddens = 128
lr = 0.0002
test_iter = 100
load_pretrained = True
logger_name = 'train'
######### Set Seeds ###########
save_path =  os.path.join('model_zoo',str(sigma))
if not os.path.exists(save_path):
    os.makedirs(save_path)
utils_logger.logger_info(logger_name, os.path.join('model_zoo',str(sigma), logger_name + '.log'))
logger = logging.getLogger(logger_name)

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
            noisy_flip = torch.rot90(noise, k=3, dims=[-1, -2])  # 270度
            noisy_rot = torch.rot90(noise, k=1, dims=[-1, -2])
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
            count += 1
    avg_psnr = avg_psnr / count
    return avg_psnr, psnrs


######### Model ###########
if __name__ == '__main__':
    total_loss_logger = VisdomPlotLogger('line', opts={'title': 'Real Loss'})
    psnr_logger = VisdomPlotLogger('line', opts={'title': 'Train PSNR'})

    mhcnn = Net(in_channels,in_channels,hiddens).cuda()
    mhcnn = torch.nn.DataParallel(mhcnn)
    if load_pretrained:
        mhcnn.load_state_dict(torch.load(os.path.join(save_path,'best.pth')))
    print('load success')

    ######### Scheduler ###########
    total_lossfn = nn.L1Loss().cuda()
    noise_net_lossfn = nn.MSELoss().cuda()  # 不行的话用EM距离 Wasserstein loss
    optimizer = torch.optim.Adam(mhcnn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-7)

    ######### DataLoaders ###########
    img_options_train = {'patch_size': patch_sizes}
    train_dataset = get_training_data(train_dir, img_options_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                              drop_last=False)
    best_psnr = 0
    best_epoch = 0
    best_iter = 0

    print("Evaluation after every {" + str(test_iter) + "} Iterations !!!\n")
    all_noisy_imgs = scipy.io.loadmat(r'E:\image_denoising\zzz-finished\DRNet\DRNet\DR_new\testsets\ValidationNoisyBlocksSrgb.mat')[
    'ValidationNoisyBlocksSrgb']
    all_clean_imgs = scipy.io.loadmat(r'E:\image_denoising\zzz-finished\DRNet\DRNet\DR_new\testsets\ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb']

    best = 0
    count = 0
    for epo in range(100000):
        print('epoch',epo)
        for i, data in enumerate(train_loader):
            count += 1
            # print(count)
            noisy, noisy_flip, noisy_rot, clean = data
            noisy, noisy_flip, noisy_rot, clean = noisy.cuda(), noisy_flip.cuda(), noisy_rot.cuda(), clean.cuda()
            out = mhcnn(noisy,noisy_flip,noisy_rot)
            loss = total_lossfn(out, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if count%test_iter == 0:
                print('now iter', count)
                total_loss_logger.log(count//test_iter, float(loss))
                test_val_, psnrs = test(mhcnn)
                psnr_logger.log(count//test_iter, float(test_val_))


                if test_val_ > best:
                    torch.save(mhcnn.state_dict(), os.path.join(save_path,'best.pth'))
                    best = test_val_
                    print('now_best',best)
                    logger.info('best PSNR:[{}] iter:[{}]'.format(test_val_, count))
                    # for i_psnr in range(len(psnrs)):
                    #     logger.info('{:->4d}| {:<4.2f}dB'.format(i_psnr, psnrs[i_psnr]))
                    # logger.info('---------------')
                mhcnn.train()
        if epo % 3 == 0 and epo > 0:
            scheduler.step()
            print('epoch', epo, ' current learning rate', optimizer.param_groups[0]['lr'])

