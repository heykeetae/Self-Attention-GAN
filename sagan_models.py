import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np


class Self_Attn(nn.Module):
    '''
    input: batch_size x feature_depth x feature_size x feature_size
    attn_score: batch_size x feature_size x feature_size
    output: batch_size x feature_depth x feature_size x feature_size
    '''

    def __init__(self, b_size, imsize, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.b_size = b_size
        self.imsize = imsize
        self.in_dim = in_dim
        self.f_ = nn.Conv2d(in_dim, int(in_dim / 8)*(imsize ** 2), 1)
        self.g_ = nn.Conv2d(in_dim, int(in_dim / 8)*(imsize ** 2), 1)
        self.h_ = nn.Conv2d(in_dim, in_dim, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, pixel_wise=True):
        if pixel_wise:
            b_size = x.size(0)
            f_size = x.size(-1)

            f_x = self.f_(x)  # batch x in_dim/8*f*f x f_size x f_size
            g_x = self.g_(x)  # batch x in_dim/8*f*f x f_size x f_size
            h_x = self.h_(x).unsqueeze(2).repeat(1, 1, f_size ** 2, 1, 1).contiguous()\
                                                                .view(b_size, -1,f_size ** 2,f_size ** 2)  # batch x in_dim x f*f x f_size x f_size

            f_ready = f_x.contiguous().view(b_size, -1, f_size ** 2, f_size, f_size).permute(0, 1, 2, 4,
                                                                                             3)  # batch x in_dim*f*f x f_size2 x f_size1
            g_ready = g_x.contiguous().view(b_size, -1, f_size ** 2, f_size,
                                            f_size)  # batch x in_dim*f*f x f_size1 x f_size2

            attn_dist = torch.mul(f_ready, g_ready).sum(dim=1).contiguous().view(-1, f_size ** 2)  # batch*f*f x f_size1*f_size2
            attn_soft = F.softmax(attn_dist, dim=1).contiguous().view(b_size, f_size ** 2, f_size ** 2)  # batch x f*f x f*f
            attn_score = attn_soft.unsqueeze(1)  # batch x 1 x f*f x f*f

            self_attn_map = torch.mul(h_x, attn_score).sum(dim=3).contiguous().view(b_size, -1, f_size, f_size)  # batch x in_dim x f*f

            self_attn_map = self.gamma * self_attn_map + x

        return self_attn_map, attn_score

class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(batch_size, int(self.imsize/4), 128, 'relu')
        self.attn2 = Self_Attn(batch_size, int(self.imsize/2), 64, 'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.l1(z)
        out=self.l2(out)
        out=self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out, p1, p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(batch_size, int(self.imsize/8), 256, 'relu')
        self.attn2 = Self_Attn(batch_size, int(self.imsize/16), 512, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out.squeeze(), p1, p2
