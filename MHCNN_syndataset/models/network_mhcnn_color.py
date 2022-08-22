import torch
import torch.nn as nn
import models.basicblock as B
from models.transformer import Multi_Scale_Attention5
# 还是不好的话，可以去掉那两次注意力，把nc改成128
class D_Block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(D_Block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in / 2.), kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in * 3 / 2.), out_channels=int(channel_in / 2.), kernel_size=3,
                                stride=1, padding=1)
        self.relu2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in * 2, out_channels=channel_in, kernel_size=3, stride=1,
                                padding=1)
        self.relu3 = nn.PReLU()
        self.tail = B.conv(channel_in, channel_out, mode='CBR')
    def forward(self, x):
        residual = x
        out = self.relu1(self.conv_1(x))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.conv_2(conc))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.conv_3(conc))
        out = torch.add(out, residual)
        out = self.tail(out)
        return out

'''
网络输入为3个图像，训练时在数据集中产生，测试时需要另外产生
一个是原图，
一个是np.rot90(img, k=3)
一个是np.flipud(img)
'''
class Net(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=128, nb=17, act_mode='BR'):
        super(Net, self).__init__()
        self.conv1 = B.conv(in_nc, nc*3, 1, 1, 0, mode='C')
        self.path1 = nn.Sequential(
            B.conv(in_nc, nc, 1, 1, 0, mode='C'),
            D_Block(nc, nc),
            D_Block(nc,nc)
        )
        self.path2 = nn.Sequential(
            B.conv(in_nc, nc, 1, 1, 0, mode='C'),
            D_Block(nc, nc),
            D_Block(nc,nc)
        )
        self.path3 = nn.Sequential(
            B.conv(in_nc, nc, 1, 1, 0, mode='C'),
            D_Block(nc, nc),
            D_Block(nc,nc)
        )

        self.msa = Multi_Scale_Attention5(nc)
        self.channel_att = B.eca_layer(nc*3)

        tail = [
                D_Block(nc * 3, nc*3),
                D_Block(nc * 3, nc * 3),
                D_Block(nc * 3, nc * 3),
                D_Block(nc*3, nc),
                B.conv(nc, out_nc, 1, 1, 0, mode='C')
                ]
        self.tail = B.sequential(*tail)

    def forward(self, x, xf, xr): # residual
        residual = x
        xs = self.conv1(x)
        x = self.path1(x)
        xf = self.path2(xf)
        xr = self.path3(xr)

        x = self.msa(x,xf,xr)
        x = xs - x

        x = self.channel_att(x)
        out = self.tail(x)
        out = residual - out
        return out

if __name__ == '__main__':
    net = Net()
    print(net)