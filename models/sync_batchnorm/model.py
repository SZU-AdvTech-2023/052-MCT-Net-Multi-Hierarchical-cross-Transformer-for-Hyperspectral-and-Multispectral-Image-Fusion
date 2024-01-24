import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EDSR(nn.Module):
    def __init__(self, n_resblocks, n_feats, upscale_factor, n_colors, res_scale, conv=default_conv):
        super(EDSR, self).__init__()
        scale = upscale_factor
        kernel_size = 3
        act = nn.ReLU(True)
        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x


class my_model(nn.Module):
    def __init__(self, Ch, stages, sf):  # ch：通道数，stages：迭代次数默认4，sf：放大倍数
        super(my_model, self).__init__()
        self.Ch = Ch
        self.s = stages
        self.sf = sf

        ## The modules for learning the measurement matrix R and R^T
        self.RT = nn.Sequential(nn.Conv2d(3, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
        self.R = nn.Sequential(nn.Conv2d(Ch, 3, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())

        ## The modules for learning the measurement matrix B and B^T
        if self.sf == 8:
            self.BT = nn.Sequential(nn.ConvTranspose2d(Ch, Ch, kernel_size=12, stride=8, padding=2), nn.LeakyReLU())
            self.B = nn.Sequential(nn.Conv2d(Ch, Ch, kernel_size=12, stride=8, padding=2), nn.LeakyReLU())
        elif self.sf == 16:
            self.BT = nn.Sequential(nn.ConvTranspose2d(Ch, Ch, kernel_size=6, stride=4, padding=1),
                                    nn.LeakyReLU(),
                                    nn.ConvTranspose2d(Ch, Ch, kernel_size=6, stride=4, padding=1), nn.LeakyReLU())
            self.B = nn.Sequential(nn.Conv2d(Ch, Ch, kernel_size=6, stride=4, padding=1),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(Ch, Ch, kernel_size=6, stride=4, padding=1), nn.LeakyReLU())

        self.conv = nn.Conv2d(Ch + 3, 64, kernel_size=3, stride=1, padding=1)

        ## Dense connection
        self.Den_con1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.Den_con2 = nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1)
        self.Den_con3 = nn.Conv2d(64 * 3, 64, kernel_size=3, stride=1, padding=1)
        self.Den_con4 = nn.Conv2d(64 * 4, 64, kernel_size=3, stride=1, padding=1)

        self.lamda_0 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_0 = Parameter(torch.ones(1), requires_grad=True)
        self.lamda_1 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_1 = Parameter(torch.ones(1), requires_grad=True)
        self.lamda_2 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_2 = Parameter(torch.ones(1), requires_grad=True)
        self.lamda_3 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_3 = Parameter(torch.ones(1), requires_grad=True)

        self._initialize_weights()
        torch.nn.init.normal_(self.lamda_0, mean=1, std=0.01)
        torch.nn.init.normal_(self.eta_0, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lamda_1, mean=1, std=0.01)
        torch.nn.init.normal_(self.eta_1, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lamda_2, mean=1, std=0.01)
        torch.nn.init.normal_(self.eta_2, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.lamda_3, mean=1, std=0.01)
        torch.nn.init.normal_(self.eta_3, mean=0.1, std=0.01)

        # reconstruct_model
        self.edsr = EDSR(n_resblocks=16, n_feats=64, upscale_factor=1, n_colors=64, res_scale=1)
        self.edsr_final = nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def reconnect(self, Res1, Res2, Xt, Ut, i):
        if i == 0:
            eta = self.eta_0
            lamda = self.lamda_0
        elif i == 1:
            eta = self.eta_1
            lamda = self.lamda_1
        elif i == 2:
            eta = self.eta_2
            lamda = self.lamda_2
        elif i == 3:
            eta = self.eta_3
            lamda = self.lamda_3

        Xt = Xt - 2 * eta * (Res1 + Res2 + lamda * (Xt - Ut))
        return Xt

    def forward(self, Y, X):  # Y: RGB  X: LR
        re_list = []
        Zt = F.interpolate(X, scale_factor=self.sf, mode='bicubic', align_corners=False)  # Z^(0)
        for i in range(0, self.s):
            ZtR = self.R(Zt)
            Res1 = self.RT(ZtR - Y)
            BZt = self.B(Zt)
            Res2 = self.BT(BZt - X)
            feat = self.conv(torch.cat((Zt, Y), 1))
            if i == 0:
                re_list.append(feat)
                fufeat = self.Den_con1(feat)
            elif i == 1:
                re_list.append(feat)
                fufeat = self.Den_con2(torch.cat(re_list, 1))
            elif i == 2:
                re_list.append(feat)
                fufeat = self.Den_con3(torch.cat(re_list, 1))
            elif i == 3:
                re_list.append(feat)
                fufeat = self.Den_con4(torch.cat(re_list, 1))
            Ut = self.edsr(fufeat)
            Ut = self.edsr_final(Ut)
            Zt = self.reconnect(Res1, Res2, Zt, Ut, i)
        return Zt


if __name__ == '__main__':  # 记得注释forward第一行
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    from thop import profile

    if 0:
        x = torch.randn(1, 31, 64, 64).cuda()
        y = torch.randn(1, 3, 512, 512).cuda()
        model = MPRNet_up(Ch=31, stages=4, sf=8).cuda()
    else:
        x = torch.randn(1, 31, 64, 64).to('cpu')
        y = torch.randn(1, 3, 512, 512).to('cpu')
        model = my_model(Ch=31, stages=4, sf=8).to('cpu')
    z = model(y, x)
    print(z.shape)
    flops, params = profile(model, inputs=(y, x))
    print('flops is %.3f' % (flops / 1e11))  ## 打印计算量
    print('params is %.3f' % (params / 1e5))  ## 打印参数量
    print('end')
