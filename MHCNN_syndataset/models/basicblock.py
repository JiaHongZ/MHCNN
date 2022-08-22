from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.batchrenorm import BatchRenorm2d



def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


'''
# --------------------------------------------
# Useful blocks
# https://github.com/xinntao/BasicSR
# --------------------------------
# conv + normaliation + relu (conv)
# (PixelUnShuffle)
# (ConditionalBatchNorm2d)
# concat (ConcatBlock)
# sum (ShortcutBlock)
# resblock (ResBlock)
# Channel Attention (CA) Layer (CALayer)
# Residual Channel Attention Block (RCABlock)
# Residual Channel Attention Group (RCAGroup)
# Residual Dense Block (ResidualDenseBlock_5C)
# Residual in Residual Dense Block (RRDB)
# --------------------------------------------
'''


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
# 膨胀卷积固定成了2
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, dilation=2, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'D': #膨胀卷积
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=2, bias=bias,dilation=dilation))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == 'P':
            L.append(nn.PReLU())
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'N': #BatchReNomal, 32与batch_size一样
            L.append(BatchRenorm2d(out_channels))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


# --------------------------------------------
# inverse of pixel_shuffle
# --------------------------------------------
def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    '''Tensor.contiguous()函数不会对原始数据进行任何修改，而仅仅对其进行复制，
    并在内存空间上进行对齐，即在内存空间上，tensor元素的内存地址保持连续。'''
    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    # 对于一个高维的Tensor执行permute，我们没有改变数据的相对位置，而只是旋转了一下这个(超)立方体。或者也可以说，改变了我们对这个(超)立方体的“观察角度”而已。
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


# --------------------------------------------
# conditional batch norm
# https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
# --------------------------------------------
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


# --------------------------------------------
# Concat the output of a submodule to its input
# --------------------------------------------
class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        return self.sub.__repr__() + 'concat'


# --------------------------------------------
# sum the output of a submodule to its input
# --------------------------------------------
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr
'''
Zhang Jiahong 2021.1.18
'''
#--------------------------------
# original no-local block
# https://arxiv.org/pdf/1711.07971.pdf
#--------------------------------
class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


# --------------------------------
# no-local block
# Non-Local Recurrent Network for Image Restoration
#---------------------------------
class NonLocalBlock_NLRN(nn.Module):
    def __init__(self, channel, field_size):
        super(NonLocalBlock, self).__init__()
        self.field_size = field_size
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        # field_size 是窗口大小
        x_theta = self.conv_theta(x)
        x_g = self.conv_g(x)
        x_phi = self.conv_phi(x)

        x_theta = torch.unsqueeze(x_theta, -2)
        x_phi_patches = nn.functional.unfold(
            x_phi, [1, self.field_size, self.field_size, 1], [1, 1, 1, 1], [1, 1, 1, 1],
            padding='SAME')
        x_phi_patches = torch.reshape(x_phi_patches, [b, c, h, self.field_size * self.field_size, w]).contiguous()
        x_phi_patches = x_phi_patches.transpose()
        x_mul1 = torch.matmul(x_theta, x_phi_patches)
        x_mul1_softmax = self.softmax(x_mul1, axis=-1)

        x_g_patches = nn.functional.unfold(
            x_g, [1, self.field_size, self.field_size, 1], [1, 1, 1, 1], [1, 1, 1, 1],
            padding='SAME')
        x_g_patches = torch.reshape(x_g_patches, [b, c, h, self.field_size * self.field_size, w]).contiguous()
        x_g_patches = x_g_patches.transpose()
        x_mul2 = torch.matmul(x_mul1_softmax, x_g_patches)
        x_mul2_reshaped = torch.reshape(x_mul2, [b, c, h, w]).contiguous()

        mask = self.conv_mask(x_mul2_reshaped)
        out = mask + x
        return out
''' 
    Zhang Jiahong
    at 2020.12.16
'''
class SPBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, upFactor=2, negative_slope=0.2):
        super().__init__()
        self.conv1 = conv(in_channels=in_channels, out_channels=out_channels*upFactor*upFactor,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias,mode='CR')
        self.up = nn.PixelShuffle(upscale_factor=upFactor)
        # input is 40*40, now 80*80, the next image size is (80-4+2*1)/2 + 1= 40
        self.conv2 = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=bias,mode='CR')

    def forward(self,x):
        out = self.conv1(x)
        out = self.up(out)
        out = self.conv2(out)
        return out

class SPBlock2(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, upFactor=2,
                 negative_slope=0.2):
        super().__init__()
        self.conv1 = conv(in_channels=in_channels, out_channels=out_channels * upFactor * upFactor,
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, mode='CR')
        self.up = nn.PixelShuffle(upscale_factor=upFactor)
        # input is 40*40, now 80*80, the next image size is (80-4+2*1)/2 + 1= 40
        self.conv2 = conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                          bias=bias, mode='CBR')
        self.conv3 = conv(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1,
                          bias=bias, mode='CR')

    def forward(self, x):
        out = self.conv1(x)
        out = self.up(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, upFactor=2, negative_slope=0.2):
        super().__init__()
        self.up = nn.PixelShuffle(upscale_factor=upFactor)
        assert in_channels%4 == 0

        self.conv2 = conv(in_channels=int(in_channels/4), out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias,mode='CR')

    def forward(self,x):
        out = self.up(x)
        out = self.conv2(out)
        return out
# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias,1, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res
class ResBlock_CRD(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRD', negative_slope=0.2):
        super(ResBlock_CRD, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias,2, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res

class ResBlock_CDC(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CDC', negative_slope=0.2):
        super(ResBlock_CDC, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias,2, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res

class ResBlock_CCC(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CCC', negative_slope=0.2):
        super(ResBlock_CCC, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias,1, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res

class ResBlock_CD2C(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CCC', negative_slope=0.2):
        super(ResBlock_CD2C, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias,1, 'C', negative_slope)
        self.res2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=stride, padding=2, bias=bias, dilation=4)
        self.res3 = conv(in_channels, out_channels, kernel_size, stride, padding, bias,1, 'C', negative_slope)


    def forward(self, x):
        res = self.res1(x)
        res = self.res2(res)
        res = self.res3(res)

        return x + res
# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))    CBRCR
# --------------------------------------------
# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock_ResDNN(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CCR',
                 negative_slope=0.2):
        super(ResBlock_ResDNN, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, 1, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res
# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))    CBRDB
# --------------------------------------------
class ResBlock_ResCBRDB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBRDB',
                 negative_slope=0.2):
        super(ResBlock_ResCBRDB, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, 2, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res

# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))    CBRDB
# --------------------------------------------
class ResBlock_ResCBRCR(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBRCB',
                 negative_slope=0.2):
        super(ResBlock_ResCBRCR, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, 1, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res
# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))    CBRCB
# --------------------------------------------
class ResBlock_ResCBRCB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBRCB',
                 negative_slope=0.2):
        super(ResBlock_ResCBRCB, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, 1, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res
# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))    CBRCBR
# --------------------------------------------
class ResBlock_ResCBRCBR(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBRCBR',
                 negative_slope=0.2):
        super(ResBlock_ResCBRCBR, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, 1, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res
        # --------------------------------------------

# Res Block: x + conv(relu(conv(x)))    CBRDBR
# --------------------------------------------
class ResBlock_ResCBRDBR(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBRDBR',
                 negative_slope=0.2):
        super(ResBlock_ResCBRDBR, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding,bias, 2, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res
# Res Block: x + conv(relu(conv(x)))    CBRDR
# --------------------------------------------
class ResBlock_ResCBRDR(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBRDR',
                 negative_slope=0.2):
        super(ResBlock_ResCBRDR, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding,bias, 2, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res
# --------------------------------------------
# simplified information multi-distillation block (IMDB)
# x + conv1(concat(split(relu(conv(x)))x3))
# --------------------------------------------
class IMDBlock(nn.Module):
    """
    @inproceedings{hui2019lightweight,
      title={Lightweight Image Super-Resolution with Information Multi-distillation Network},
      author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},
      booktitle={Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)},
      pages={2024--2032},
      year={2019}
    }
    @inproceedings{zhang2019aim,
      title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
      author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
      booktitle={IEEE International Conference on Computer Vision Workshops},
      year={2019}
    }
    """
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CL', d_rate=0.25, negative_slope=0.05):
        super(IMDBlock, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        assert mode[0] == 'C', 'convolutional layer first'

        self.conv1 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(self.r_nc, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0], negative_slope=negative_slope)

    def forward(self, x):
        d1, r1 = torch.split(self.conv1(x), (self.d_nc, self.r_nc), dim=1)
        d2, r2 = torch.split(self.conv2(r1), (self.d_nc, self.r_nc), dim=1)
        d3, r3 = torch.split(self.conv3(r2), (self.d_nc, self.r_nc), dim=1)
        d4 = self.conv4(r3)
        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1))
        return x + res


# --------------------------------------------
# Enhanced Spatial Attention (ESA)
# --------------------------------------------
class ESA(nn.Module):
    def __init__(self, channel=64, reduction=4, bias=True):
        super(ESA, self).__init__()
        #               -->conv3x3(conv21)-----------------------------------------------------------------------------------------+
        # conv1x1(conv1)-->conv3x3-2(conv2)-->maxpool7-3-->conv3x3(conv3)(relu)-->conv3x3(conv4)(relu)-->conv3x3(conv5)-->bilinear--->conv1x1(conv6)-->sigmoid
        self.r_nc = channel // reduction
        self.conv1 = nn.Conv2d(channel, self.r_nc, kernel_size=1)
        self.conv21 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=1)
        self.conv2 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(self.r_nc, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = F.max_pool2d(self.conv2(x1), kernel_size=7, stride=3)  # 1/6
        x2 = self.relu(self.conv3(x2))
        x2 = self.relu(self.conv4(x2))
        x2 = F.interpolate(self.conv5(x2), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x2 = self.conv6(x2 + self.conv21(x1))
        return x.mul(self.sigmoid(x2))
        # return x.mul_(self.sigmoid(x2))


class CFRB(nn.Module):
    def __init__(self, in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1, bias=True, mode='CL', d_rate=0.5, negative_slope=0.05):
        super(CFRB, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = in_channels  # int(in_channels - self.d_nc)

        assert mode[0] == 'C', 'convolutional layer first'

        self.conv1_d = conv(in_channels, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.conv1_r = conv(in_channels, self.r_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv2_d = conv(self.r_nc, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.conv2_r = conv(self.r_nc, self.r_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv3_d = conv(self.r_nc, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.conv3_r = conv(self.r_nc, self.r_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv4_d = conv(self.r_nc, self.d_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.act = conv(mode=mode[-1], negative_slope=negative_slope)
        self.esa = ESA(in_channels, reduction=4, bias=True)

    def forward(self, x):
        d1 = self.conv1_d(x)
        x = self.act(self.conv1_r(x)+x)
        d2 = self.conv2_d(x)
        x = self.act(self.conv2_r(x)+x)
        d3 = self.conv3_d(x)
        x = self.act(self.conv3_r(x)+x)
        x = self.conv4_d(x)
        x = self.act(torch.cat([d1, d2, d3, x], dim=1))
        x = self.esa(self.conv1x1(x))
        return x


# --------------------------------------------
# Channel Attention (CA) Layer
# --------------------------------------------
class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_fc(y)
        return x * y


# --------------------------------------------
# Residual Channel Attention Block (RCAB)
# --------------------------------------------
class RCABlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', reduction=16, negative_slope=0.2):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, x):
        res = self.res(x)
        res = self.ca(res)
        return res + x


# --------------------------------------------
# Residual Channel Attention Group (RG)
# --------------------------------------------
class RCAGroup(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', reduction=16, nb=12, negative_slope=0.2):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding, bias, mode, reduction, negative_slope)  for _ in range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))
        self.rg = nn.Sequential(*RG)  # self.rg = ShortcutBlock(nn.Sequential(*RG))

    def forward(self, x):
        res = self.rg(x)
        return res + x


# --------------------------------------------
# Residual Dense Block
# style: 5 convs
# --------------------------------------------
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR', negative_slope=0.2):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel
        self.conv1 = conv(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(nc+gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(nc+2*gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(nc+3*gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv5 = conv(nc+4*gc, nc, kernel_size, stride, padding, bias, mode[:-1], negative_slope)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul_(0.2) + x


# --------------------------------------------
# Residual in Residual Dense Block
# 3x5c
# --------------------------------------------
class RRDB(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR', negative_slope=0.2):
        super(RRDB, self).__init__()

        self.RDB1 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.RDB2 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.RDB3 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul_(0.2) + x


"""
# --------------------------------------------
# Upsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# upsample_pixelshuffle
# upsample_upconv
# upsample_convtranspose
# --------------------------------------------
"""


# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    elif mode[0] == '4':
        uc = 'vC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1


'''
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
'''


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


'''
# --------------------------------------------
# NonLocalBlock2D:
# embedded_gaussian
# +W(softmax(thetaXphi)Xg)
# --------------------------------------------
'''


# --------------------------------------------
# non-local block with embedded_gaussian
# https://github.com/AlexHex7/Non-local_pytorch
# --------------------------------------------
class NonLocalBlock2D(nn.Module):
    def __init__(self, nc=64, kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='maxpool', negative_slope=0.2):

        super(NonLocalBlock2D, self).__init__()

        inter_nc = nc // 2
        self.inter_nc = inter_nc
        self.W = conv(inter_nc, nc, kernel_size, stride, padding, bias, mode='C'+act_mode)
        self.theta = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

        if downsample:
            if downsample_mode == 'avgpool':
                downsample_block = downsample_avgpool
            elif downsample_mode == 'maxpool':
                downsample_block = downsample_maxpool
            elif downsample_mode == 'strideconv':
                downsample_block = downsample_strideconv
            else:
                raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
            self.phi = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
            self.g = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
        else:
            self.phi = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')
            self.g = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_nc, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_nc, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_nc, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_nc, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

'''
 No-local original
 zhang jia hong
 2021.1.20
'''
class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out
'''
 No-local2 - NLRN
 zhang jia hong
 2021.1.20
'''
class NonLocalBlock_NLRN(nn.Module):
    def __init__(self, channel, field_size):
        super(NonLocalBlock_NLRN, self).__init__()
        self.inter_channel = channel // 2
        if channel == 1:
            self.inter_channel = 1
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.field_size = field_size
    def forward(self, x):
        import utils.utils_image as utill
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x)
        x_g = self.conv_g(x)
        x_theta = self.conv_theta(x)

        x_theta = x_theta.unsqueeze(-2) # 16x256x30x1x30
        x_phi_patches = utill.extract_image_patches(
            x_phi, [self.field_size, self.field_size], [1, 1], [1, 1],
            padding='same')   # 16x256x30x16x30
        x_phi_patches = torch.reshape(x_phi_patches, [
            x_phi.shape[0],  # N
            x_phi.shape[2],  # H
            x_phi.shape[3],  # W
            x_phi.shape[1],  # C
            self.field_size * self.field_size, # patch l
        ]).contiguous()

        x_theta = x_theta.permute(0, 2, 4, 3 ,1).contiguous()
        x_mul1 = torch.matmul(x_theta, x_phi_patches) # 16x30x30x1x16
        x_mul1_softmax = self.softmax(x_mul1)

        x_g_patches = utill.extract_image_patches(
            x_g, [self.field_size, self.field_size], [1, 1], [1, 1],
            padding='same')
        x_g_patches = torch.reshape(x_g_patches, [
            x_phi.shape[0],  # N
            x_phi.shape[2],  # H
            x_phi.shape[3],  # W
            self.field_size * self.field_size,  # patch l
            x_phi.shape[1],  # C
        ]).contiguous()       # 16x30x30x16x256(inter_channel)
        x_mul2 = torch.matmul(x_mul1_softmax, x_g_patches) # 16x30x30x1x256(inter_channel)
        x_mul2_reshaped = x_mul2.permute(0, 4, 3, 1 ,2).contiguous().view(b,self.inter_channel, h, w)

        mask = self.conv_mask(x_mul2_reshaped)
        out = mask + x
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class _DCR_block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(_DCR_block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in / 2.), kernel_size=3, stride=1,
                                padding=1)
        self.relu1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in * 3 / 2.), out_channels=int(channel_in / 2.), kernel_size=3,
                                stride=1, padding=1)
        self.relu2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in * 2, out_channels=channel_in, kernel_size=3, stride=1,
                                padding=1)
        self.relu3 = nn.PReLU()
    def forward(self, x):
        residual = x

        out = self.relu1(self.conv_1(x))

        conc = torch.cat([x, out], 1)

        out = self.relu2(self.conv_2(conc))

        conc = torch.cat([conc, out], 1)

        out = self.relu3(self.conv_3(conc))

        out = torch.add(out, residual)
        return out

# ASPP

class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()

        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)

        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

# wavelet
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


'''
    ECANet, a effective channel attention method
    2020 CVPR https://github.com/BangguWu/ECANet
'''
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channels: Number of channels in the input tensor
        b: Hyper-parameter for adaptive kernel size formulation. Default: 1
        gamma: Hyper-parameter for adaptive kernel size formulation. Default: 2 
    """
    def __init__(self, channels, b=1, gamma=2):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        import math
        k = int(abs((math.log2(self.channels)/self.gamma)+ self.b/self.gamma))
        out = k if k % 2 else k+1
        return out

    def forward(self, x):

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)