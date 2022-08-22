import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class MyDataset(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(MyDataset, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = H_path

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = np.random.randint(0, 8)
            patch_H = util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = img_H.clone()

            # --------------------------------
            # add noise
            # --------------------------------
            noise = torch.randn(img_L.size()).mul_(self.sigma/255.0)
            img_L.add_(noise)

            img_L_flip = img_L.clone()
            img_L_rot = img_L.clone()
            img_L_flip = torch.rot90(img_L_flip, k=3, dims=[-1,-2]) # 270度
            img_L_rot = torch.rot90(img_L_rot, k=2, dims=[-1,-2])


            # import matplotlib.pyplot as plt
            # # img = np.array(img_L_rot)[0]
            # # print(img.shape)
            # # plt.imshow(img)
            # img = np.array(img_L)[0]
            # print(img.shape)
            # plt.imshow(img)
            # plt.show()

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)

            # --------------------------------
            # add noise
            # --------------------------------
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test/255.0, img_L.shape)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_L = util.single2tensor3(img_L)
            img_H = util.single2tensor3(img_H)


            img_L_flip = img_L.clone()
            img_L_rot = img_L.clone()
            img_L_flip = torch.rot90(img_L_flip, k=3, dims=[-1,-2]) # 270度
            img_L_rot = torch.rot90(img_L_rot, k=2, dims=[-1,-2])
        return {'L': img_L, 'H': img_H,
                'LF': img_L_flip,'LR': img_L_rot,
                'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)
