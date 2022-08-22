"""
Adapted from https://github.com/lukemelas/simple-bert
"""
 
import numpy as np
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import models.basicblock as B
# 展开，linear不行，因为你图像尺寸是变得，所以得用1x1卷积作为算子
class Multi_Scale_Attention_old(nn.Module): # 输入3个 b c h w，输出 1 个 b c h w
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, dim,  dropout = 0):
        super().__init__()
        self.normx = nn.LayerNorm(dim, eps=1e-6)
        self.normxf = nn.LayerNorm(dim, eps=1e-6)
        self.normxr = nn.LayerNorm(dim, eps=1e-6)
        self.normx2 = nn.LayerNorm(dim, eps=1e-6)

        self.proj_x1 = nn.Linear(dim, dim)
        self.proj_x2 = nn.Linear(dim, dim)
        self.proj_xf = nn.Linear(dim, dim)
        self.proj_xr = nn.Linear(dim, dim)
        self.proj_scores = nn.Linear(dim, dim)

        self.drop = nn.Dropout(dropout)

    def forward(self, x,xf,xr):
        b,c,h,w = x.shape
        x = torch.flatten(x,start_dim=1) # b c h*w
        xf = torch.flatten(xf,start_dim=1) # b c h*w
        xr = torch.flatten(xr,start_dim=1) # b c h*w
        print('x_msa',x.shape)
        x1 = self.proj_x1(self.normx(x))
        x2 = self.proj_x2(self.normx(x))
        xf = self.proj_xf(self.normxf(xf))
        xr = self.proj_xr(self.normxr(xr))

        x1 = rearrange(x1, 'b c h*w -> b h c w')
        x2 = rearrange(x2, 'b c h*w -> b h c w')
        xf = rearrange(xf, 'b c h*w -> b h w c')
        xr = rearrange(xr, 'b c h*w -> b h w c')
        score1 = x1 @ xf / np.sqrt(xf.size(-1)) # b h c c
        score2 = x2 @ xr / np.sqrt(xr.size(-1))
        scores = score1 + score2
        scores = self.drop(self.proj_scores(F.softmax(scores, dim=-1)))# b h c c

        scores = rearrange(scores, 'b h c c -> b c h c')
        x = rearrange(x1, 'b c h*w -> b h c w')

        x = (scores @ x).transpose(1, 2).contiguous() # b c h w
        x = self.drop(self.normx2(x))

        return x
class Multi_Scale_Attention(nn.Module): # 输入3个 b c h w，输出 1 个 b c h w
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, dim,  dropout = 0):
        super().__init__()
        # self.normx = nn.BatchNorm2d(dim)
        # self.normxf = nn.BatchNorm2d(dim)
        # self.normxr = nn.BatchNorm2d(dim)
        # self.normx2 = nn.BatchNorm2d(dim)
        self.normx = nn.InstanceNorm2d(dim)
        self.normxf = nn.InstanceNorm2d(dim)
        self.normxr = nn.InstanceNorm2d(dim)
        self.normx2 = nn.InstanceNorm2d(dim)

        self.proj_x1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_x2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_xf = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_xr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_scores = nn.Conv2d(dim, dim, 1, 1, 0)

        self.drop = nn.Dropout(dropout)

    def forward(self, x,xf,xr):
        b,c,h,w = x.shape
        # print('x_msa',x.shape)
        x1 = self.proj_x1(self.normx(x))
        x2 = self.proj_x2(self.normx(x))
        xf = self.proj_xf(self.normxf(xf))
        xr = self.proj_xr(self.normxr(xr))

        # x1 = self.proj_x1(x)
        # x2 = self.proj_x2(x)
        # xf = self.proj_xf(xf)
        # xr = self.proj_xr(xr)

        x1 = rearrange(x1, 'b c h w -> b h c w')
        x2 = rearrange(x2, 'b c h w -> b h c w')
        xf = rearrange(xf, 'b c h w -> b h w c')
        xr = rearrange(xr, 'b c h w -> b h w c')

        # print('shape',x.shape,xf.shape,xr.shape)
        score1 = x1 @ xf / np.sqrt(xf.size(-1)) # b h c c
        score2 = x2 @ xr / np.sqrt(xr.size(-1))
        scores = score1 + score2
        scores = self.drop(F.softmax(scores, dim=-1))# b h c c

        scores = rearrange(scores, 'b h c1 c2 -> b h c1 c2')
        x = rearrange(x, 'b c h w -> b h c w')
        x = scores @ x / np.sqrt(x.size(-1)) # b h c w
        x = rearrange(x, 'b h c w -> b c h w')
        x = self.drop(self.proj_scores(self.normx2(x)))

        return x

class Multi_Scale_Attention2(nn.Module): # 输入3个 b c h w，输出 1 个 b c h w
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, dim,  dropout = 0):
        super().__init__()

        self.proj_x1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_x2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_xf = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_xr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_scores = nn.Conv2d(dim, dim, 1, 1, 0)

        self.drop = nn.Dropout(dropout)
        self.msa_conv = B.conv(dim*3, dim*3, mode='CBR')
        self.channel_att = B.eca_layer(dim*3)
    def forward(self, x,xf,xr):
        b,c,h,w = x.shape
        x_org = x
        xf_org = xf
        xr_org = xr
        # print('x_msa',x.shape)
        x1 = self.proj_x1(x)
        xf = self.proj_xf(xf)
        xr = self.proj_xr(xr)

        x1 = rearrange(x1, 'b c h w -> b h c w')
        xf = rearrange(xf, 'b c h w -> b h w c')
        xr = rearrange(xr, 'b c h w -> b h c w')

        # print('shape',x.shape,xf.shape,xr.shape)
        score1 = x1 @ xf / np.sqrt(xf.size(-1)) # b h c c
        score2 = score1 @ xr / np.sqrt(xr.size(-1)) # b h c w
        score2 = rearrange(score2, 'b h c w -> b c h w')
        score = self.drop(F.softmax(score2, dim=-1))# 得到三者的相似度

        x = torch.mul(score,x_org)
        xf = torch.mul(score,xf_org)
        xr = torch.mul(score,xr_org)

        out = torch.cat([x,xf,xr],1)
        out = self.msa_conv(out)
        out = self.channel_att(out)
        return out

class Multi_Scale_Attention4(nn.Module): # 输入3个 b c h w，输出 1 个 b c h w
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, dim,  dropout = 0):
        super().__init__()

        # self.proj_x = nn.Conv2d(dim, dim, 1, 1, 0)
        # self.proj_x1 = nn.Conv2d(dim, dim, 1, 1, 0)
        # self.proj_x2 = nn.Conv2d(dim, dim, 1, 1, 0)
        # self.proj_xf = nn.Conv2d(dim, dim, 1, 1, 0)
        # self.proj_xr = nn.Conv2d(dim, dim, 1, 1, 0)
        # self.proj_scores = nn.Conv2d(dim, dim, 1, 1, 0)

        self.proj_x = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_x1 = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_x2 = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_xf = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_xr = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_scores = B.conv(dim,dim,1,1,0, mode='C')

        self.drop = nn.Dropout(dropout)
        self.msa_conv = B.conv(dim*3, dim*3, mode='CBR')
    def forward(self, x,xf,xr):
        b,c,h,w = x.shape
        x_org = x
        xf_org = xf
        xr_org = xr
        # print('x_msa',x.shape)
        x = self.proj_x(x)
        x1 = self.proj_x1(x)
        x2 = self.proj_x2(x)
        xf = self.proj_xf(xf)
        xr = self.proj_xr(xr)


        x1 = rearrange(x1, 'b c h w -> b h c w')
        x2 = rearrange(x2, 'b c h w -> b h c w')
        xf = rearrange(xf, 'b c h w -> b h w c')
        xr = rearrange(xr, 'b c h w -> b h w c')
        x = rearrange(x, 'b c h w -> b h c w')

        # print('shape',x.shape,xf.shape,xr.shape)
        score1 = x1 @ xf
        feature_xf = score1 @ x

        score2 = x2 @ xr
        feature_xr = score2 @ x

        feature_xf = rearrange(feature_xf, 'b h c w -> b c h w')
        feature_xr = rearrange(feature_xr, 'b h c w -> b c h w')

        out = torch.cat([feature_xf,feature_xr,x_org],1)
        out = self.msa_conv(out)
        return out

# class Multi_Scale_Attention5(nn.Module): # 输入3个 b c h w，输出 1 个 b c h w
#     """Transformer with Self-Attentive Blocks"""
#     def __init__(self, dim,  dropout = 0):
#         super().__init__()
#         self.proj_x = B.conv(dim,dim,1,1,0, mode='C')
#         self.proj_x1 = B.conv(dim,dim,1,1,0, mode='C')
#         self.proj_x2 = B.conv(dim,dim,1,1,0, mode='C')
#         self.proj_xf = B.conv(dim,dim,1,1,0, mode='C')
#         self.proj_xr = B.conv(dim,dim,1,1,0, mode='C')
#         self.proj_scores = B.conv(dim,dim,1,1,0, mode='C')
#
#         self.proj_xf2 = B.conv(dim,dim,1,1,0, mode='C')
#         self.proj_xr2 = B.conv(dim,dim,1,1,0, mode='C')
#
#         self.normxf = nn.InstanceNorm2d(dim)
#         self.normxr = nn.InstanceNorm2d(dim)
#         self.normxf2 = nn.InstanceNorm2d(dim)
#         self.normxr2 = nn.InstanceNorm2d(dim)
#
#         self.drop = nn.Dropout(dropout)
#         self.msa_conv = B.conv(dim*3, dim*3, mode='CBR')
#     def forward(self, x,xf,xr):
#         b,c,h,w = x.shape
#         x_org = x
#         xf_org = xf
#         xr_org = xr
#         # print('x_msa',x.shape)
#         x = self.proj_x(x)
#         x1 = self.proj_x1(x)
#         x2 = self.proj_x2(x)
#         xf = self.proj_xf(xf)
#         xr = self.proj_xr(xr)
#
#
#         x1 = rearrange(x1, 'b c h w -> b h c w')
#         x2 = rearrange(x2, 'b c h w -> b h c w')
#         xf = rearrange(xf, 'b c h w -> b h w c')
#         xr = rearrange(xr, 'b c h w -> b h w c')
#         x = rearrange(x, 'b c h w -> b h c w')
#
#         # print('shape',x.shape,xf.shape,xr.shape)
#         score1 = x1 @ xf
#         score1 = self.normxf(score1)
#         feature_xf = score1 @ x
#         feature_xf = self.proj_xf2(feature_xf)
#         feature_xf = self.normxf2(feature_xf)
#
#         score2 = x2 @ xr
#         score2 = self.normxr(score2)
#         feature_xr = score2 @ x
#         feature_xr = self.proj_xr2(feature_xr)
#         feature_xr = self.normxr2(feature_xr)
#
#         feature_xf = rearrange(feature_xf, 'b h c w -> b c h w')
#         feature_xr = rearrange(feature_xr, 'b h c w -> b c h w')
#
#         out = torch.cat([feature_xf,feature_xr,x_org],1)
#         out = self.msa_conv(out)
#         return out
class Multi_Scale_Attention5(nn.Module): # 输入3个 b c h w，输出 1 个 b c h w
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, dim,  dropout = 0):
        super().__init__()
        self.proj_x = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_x1 = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_x2 = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_xf = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_xr = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_scores = B.conv(dim,dim,1,1,0, mode='C')

        self.normxf = nn.InstanceNorm2d(dim)
        self.normxr = nn.InstanceNorm2d(dim)
        self.normxf2 = nn.InstanceNorm2d(dim)
        self.normxr2 = nn.InstanceNorm2d(dim)

        self.drop = nn.Dropout(dropout)
        self.msa_conv = B.conv(dim*3, dim*3, mode='CBR')
    def forward(self, x,xf,xr):
        b,c,h,w = x.shape
        x_org = x
        # print('x_msa',x.shape)
        x = self.proj_x(x)
        x1 = self.proj_x1(x)
        x2 = self.proj_x2(x)
        xf = self.proj_xf(xf)
        xr = self.proj_xr(xr)

        x1 = rearrange(x1, 'b c h w -> b h c w')
        x2 = rearrange(x2, 'b c h w -> b h c w')
        xf = rearrange(xf, 'b c h w -> b h w c')
        xr = rearrange(xr, 'b c h w -> b h w c')
        x = rearrange(x, 'b c h w -> b h c w')

        # print('shape',x.shape,xf.shape,xr.shape)
        score1 = x1 @ xf
        score1 = self.normxf(score1)
        feature_xf = score1 @ x
        feature_xf = self.normxf2(feature_xf)

        score2 = x2 @ xr
        score2 = self.normxr(score2)
        feature_xr = score2 @ x
        feature_xr = self.normxr2(feature_xr)

        feature_xf = rearrange(feature_xf, 'b h c w -> b c h w')
        feature_xr = rearrange(feature_xr, 'b h c w -> b c h w')

        out = torch.cat([feature_xf,feature_xr,x_org],1)
        out = self.msa_conv(out)
        return out
class Multi_Scale_Attention6(nn.Module): # 输入3个 b c h w，输出 1 个 b c h w
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, dim,  dropout = 0):
        super().__init__()
        self.proj_x = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_x1 = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_x2 = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_x3 = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_xf = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_xr = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_xr2 = B.conv(dim,dim,1,1,0, mode='C')
        self.proj_scores = B.conv(dim,dim,1,1,0, mode='C')

        self.normxf = nn.InstanceNorm2d(dim)
        self.normxr = nn.InstanceNorm2d(dim)
        self.normxr_2 = nn.InstanceNorm2d(dim)
        self.normxf2 = nn.InstanceNorm2d(dim)
        self.normxr2 = nn.InstanceNorm2d(dim)
        self.normxr_22 = nn.InstanceNorm2d(dim)

        self.drop = nn.Dropout(dropout)
        self.msa_conv = B.conv(dim*4, dim*4, mode='CBR')
    def forward(self, x,xf,xr,xr2):
        b,c,h,w = x.shape
        x_org = x
        # print('x_msa',x.shape)
        x = self.proj_x(x)
        x1 = self.proj_x1(x)
        x2 = self.proj_x2(x)
        x3 = self.proj_x3(x)
        xf = self.proj_xf(xf)
        xr = self.proj_xr(xr)
        xr2 = self.proj_xr2(xr)


        x1 = rearrange(x1, 'b c h w -> b h c w')
        x2 = rearrange(x2, 'b c h w -> b h c w')
        x3 = rearrange(x3, 'b c h w -> b h c w')
        xf = rearrange(xf, 'b c h w -> b h w c')
        xr = rearrange(xr, 'b c h w -> b h w c')
        xr2 = rearrange(xr2, 'b c h w -> b h w c')
        x = rearrange(x, 'b c h w -> b h c w')

        # print('shape',x.shape,xf.shape,xr.shape)
        score1 = x1 @ xf
        score1 = self.normxf(score1)
        feature_xf = score1 @ x
        feature_xf = self.normxf2(feature_xf)

        score2 = x2 @ xr
        score2 = self.normxr(score2)
        feature_xr = score2 @ x
        feature_xr = self.normxr2(feature_xr)

        score3 = x3 @ xr2
        score3 = self.normxr_2(score3)
        feature_xr2 = score3 @ x
        feature_xr2 = self.normxr_22(feature_xr2)

        feature_xf = rearrange(feature_xf, 'b h c w -> b c h w')
        feature_xr = rearrange(feature_xr, 'b h c w -> b c h w')
        feature_xr2 = rearrange(feature_xr2, 'b h c w -> b c h w')

        out = torch.cat([feature_xf,feature_xr,feature_xr2,x_org],1)
        out = self.msa_conv(out)
        return out