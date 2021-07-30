#encoding:utf-8
from torch.nn import functional as F
import math
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
import cv2
import sys
# import matplotlib.pyplot as plt
# np.set_printoptions(threshold=np.inf)
# np.set_printoptions(threshold=np.nan)
# ####################################IHT########################################################
def hough_transform(rows, cols, theta_res, rho_res, padding=0, if_sparse=False):

    if padding!=0:
        # theta = np.linspace(0, 180.0, np.ceil(180.0 / theta_res) + 1.0)
        # theta = theta[0:len(theta) - 1]
        # theta = np.insert(theta, obj=0, values=-theta_res, axis=0)
        # theta = np.insert(theta, obj=len(theta), values=180.0, axis=0)
        # print('theta', theta.shape, theta)

        theta = np.linspace(0-padding*theta_res, 180.0+padding*theta_res, int(np.ceil(180.0 / theta_res) + 1.0+2*padding))
        theta = theta[0:len(theta) - 1]
        # print('theta', theta.shape, theta)


    else:
        theta = np.linspace(0, 180.0, int(np.ceil(180.0 / theta_res) + 1.0))
        theta = theta[0:len(theta) - 1]
        # print('theta', theta.shape, theta)

    D = np.sqrt((rows - 1) ** 2 + (cols - 1) ** 2)
    q = np.ceil(D / rho_res)
    nrho = 2 * q + 1
    rho = np.linspace(-q * rho_res, q * rho_res, int(nrho))
    # print('rho', rho.shape, np.where(rho==0), rho)

    w = np.size(theta)
    h = np.size(rho)
    theta = torch.from_numpy(theta)
    cos_value = torch.cos(theta * np.pi / 180.0).type(torch.FloatTensor)
    sin_value = torch.sin(theta * np.pi / 180.0).type(torch.FloatTensor)
    coding = torch.cat((sin_value, cos_value), 0).resize_(2, theta.size(0))

    ###################################################
    coords = torch.ones(rows, cols).nonzero().float()
    coords[:,0] = rows-coords[:,0]-rows//2
    coords[:,1] = coords[:,1] +1 - cols//2

    # coords2 = coords = torch.ones(rows, cols).nonzero().float()
    # coords2[:,0] = rows-coords[:,0]
    # coords2[:,1] = coords[:,1] +1
    # print('coords2', coords2, coords.size())


    vote_map = torch.mm(coords, coding).type(torch.FloatTensor)
    # print('vote_map', vote_map, vote_map.size())
    vote_index = torch.zeros(rows * cols, np.size(rho), theta.size()[0]).type(torch.FloatTensor)
    for i in range(coords.size()[0]):
        for j in range(coding.size()[1]):
            rhoVal = vote_map[i, j].numpy()
            rhoIdx = np.nonzero(np.abs(rho - rhoVal) == np.min(np.abs(rho - rhoVal)))[0]
            vote_map[i, j] = float(rhoIdx[0])
            vote_index[i, rhoIdx[0], j] = 1
    vote_rho_idx = vote_index.view(rows * cols, -1).sum(dim=0).view(h,w).sum(dim=1).gt(0.0)


    # ########## make sure the len_y is even!!!
    # center_y =np.asscalar(np.where(rho==0)[0])
    # print('center_y', center_y, center_y)
    #
    # len_y = vote_rho_idx.long().sum().item()
    # if len_y %2==0:
    #     vote_rho_idx[center_y:center_y+len_y//2]=1
    #     vote_rho_idx[center_y-len_y//2 : center_y]=1

    vote_index = vote_index[:,vote_rho_idx ,:]

    # rho_selected = rho[np.nonzero(vote_rho_idx)]
    # print('vote_rho_idx', vote_rho_idx, np.nonzero(vote_rho_idx))
    # # print('vote_rho_idx', rho[np.nonzero(vote_rho_idx)])
    # print('rho_selected', rho_selected.shape,  np.where(rho_selected==0), rho_selected)
    if if_sparse:
        vote_index_full = vote_index.view(rows * cols, -1).sum(dim=0).nonzero()
        vote_index_sparse = vote_index.view(rows * cols, -1).index_select(dim=1, index=vote_index_full.view(-1))
        return vote_index_sparse
    else:
        return vote_index.view(rows, cols, -1, w)


# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
def make_conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
    layers = []
    layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]
    # layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def nms(heat, kernel_size):
    padding = tuple((p-1)//2 for p in kernel_size)
    hmax = F.max_pool2d(heat, kernel_size, stride=1, padding=padding)
    keep = (hmax == heat) .float()
    return keep*heat

####################################################################################################################
class HT(nn.Module):
    def __init__(self, vote_index):
        super(HT, self).__init__()
        self.rows, self.cols, self.h, self.w = vote_index.size()
        self.vote_index = vote_index.view(self.rows*self.cols, self.h* self.w)
        self.total = vote_index.sum(0).max()

    def forward(self, image):
        batch = image.size(0)
        channel = image.size(1)
        image = image.view(batch,channel, -1).view(batch*channel, -1)
        image = F.relu(image)
        HT_map = image @ self.vote_index
        # HT_map = HT_map/self.total
        HT_map = HT_map/(self.cols)
        # HT_map = F.normalize(HT_map, p=float('inf'), dim=1)
        HT_map = HT_map.view(batch, channel, -1).view(batch, channel, self.h, self.w)
        return HT_map


class IHT(nn.Module):
    def __init__(self, vote_index):
        super(IHT, self).__init__()
        self.rows, self.cols, self.h, self.w = vote_index.size()
        self.vote_index = vote_index.view(self.rows*self.cols, self.h* self.w).t()

    def forward(self, input_HT):
        batch = input_HT.size(0)
        channel = input_HT.size(1)
        input_HT = F.relu(input_HT)
        input_HT = input_HT.view(batch, channel, self.h * self.w).view(batch * channel, self.h * self.w)
        # input_HT = F.normalize(input_HT, p=float("inf"), dim=1)
        IHT_map = input_HT @ self.vote_index
        IHT_map = IHT_map.view(batch, channel, -1).view(batch, channel, self.rows, self.cols)
        # return IHT_map/float(self.w)
        return IHT_map


class HTIHT(nn.Module):

    def __init__(self, vote_index, inplanes, outplanes):
        super(HTIHT, self).__init__()

        self.conv1 = nn.Sequential(*make_conv_block(inplanes, inplanes, kernel_size=(9,1), padding=(4,0), bias=True, groups=inplanes))
        self.conv2 = nn.Sequential(*make_conv_block(inplanes, outplanes, kernel_size=(9,1), padding=(4,0), bias=True, groups=1))
        self.conv3 = nn.Sequential(*make_conv_block(outplanes, outplanes, kernel_size=(9,1), padding=(4,0), bias=True, groups=1))

        self.relu = nn.ReLU(inplace=True)
        self.ht = HT(vote_index)
        self.iht = IHT(vote_index)

        filtersize = 4
        x = np.zeros(shape=((2 * filtersize + 1)))
        x[filtersize] = 1
        z = []
        for _ in range(0, inplanes):
            sigma = np.random.uniform(low=1, high=2.5, size=(1))
            y = ndimage.filters.gaussian_filter(x, sigma=sigma, order=2)
            y = -y / np.sum(np.abs(y))
            # print('y', sigma, _, inplanes)
            z.append(y)
        z = np.stack(z)
        self.conv1[0].weight.data.copy_(torch.from_numpy(z).unsqueeze(1).unsqueeze(3))
        # print('weight size', self.conv1[0].weight.data.size(), self.conv2[0].weight.size(), self.conv3[0].weight.size())
        nn.init.kaiming_normal_(self.conv2[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1[0].bias)
        nn.init.zeros_(self.conv2[0].bias)
        nn.init.zeros_(self.conv3[0].bias)


    def forward(self, x, **kwargs):

        ht = self.ht(x)
        ht_conv = self.conv1(ht)
        ht_conv = self.conv2(ht_conv)
        ht_conv = self.conv3(ht_conv)
        out = self.iht(ht_conv)

        # out = self.ht(x)
        # ht = out
        # out = self.conv1(out)
        # ht1 = out
        # out = self.conv2(out)
        # ht2 = out
        # # out = self.conv3(out)
        # # ht3 = out
        # out = self.iht(out)
        # # #
        # filtersize = 4
        # y0 = np.zeros(shape=((2 * filtersize + 1)))
        # y0[filtersize] = 1
        # y1 = ndimage.filters.gaussian_filter(y0, sigma=1, order=2)
        # y1 = -y1 / np.sum(np.abs(y1))
        #
        # y2 = ndimage.filters.gaussian_filter(y0, sigma=2.5, order=2)
        # y2 = -y2 / np.sum(np.abs(y2))
        #
        # weight1 = self.conv1[0].weight.detach()
        # weight2 = self.conv2[0].weight.detach()
        # # weight3 = self.conv3[0].weight.detach()
        # print('weight', self.conv1[0].weight.size(), self.conv2[0].weight.size())
        # #
        # import matplotlib.pyplot as plt
        # for kk in range(0, 16):
        #     # print('weight', weight1[kk,0,:,0], weight2[0,kk,:,0])
        #     # print('bn', self.conv1[1].weight[kk], self.conv2[1].weight[0])
        #     # print('bn bias', self.conv1[1].bias[0], self.conv2[1].bias[0])
        #
        #     # kk =np.random.randint(low=0, high=128, size=(1)).item()
        #     fig, axs = plt.subplots(nrows=1, ncols=5)
        #     axs = axs.ravel()
        #     ax = axs[0]
        #     ax.imshow(x[0, kk].detach(), cmap='gray')
        #     ax.set_title(str(x[0, kk].detach().max()))
        #     ax = axs[1]
        #     ax.imshow(ht[0, kk].detach(), cmap='gray')
        #     ax.set_title(str(ht[0, kk].detach().max()))
        #     ax = axs[2]
        #     ax.imshow(ht1[0, kk].detach(), cmap='gray')
        #     ax.set_title(str(ht1[0, kk].detach().max()))
        #     ax = axs[3]
        #     ax.imshow(ht2[0, kk].detach(), cmap='gray')
        #     ax.set_title(ht2[0, kk].detach().max().item())
        #     # ax = axs[4]
        #     # ax.imshow(ht3[0, kk].detach(), cmap='gray')
        #     # ax.set_title(str(ht3[0, kk].detach().max()))
        #     ax = axs[4]
        #     ax.imshow(out[0, kk].detach(), cmap='gray')
        #     ax.set_title(str(out[0, kk].detach().max()))
        #     # ax.plot(weight1[kk,0,:,0])
        #     # ax.plot(weight2[0,kk,:,0])
        #     # ax.plot(y1, '--')
        #     # ax.plot(y2, '-.')
        #     plt.suptitle(str(kk))
        #     plt.show()
        return ht*self.ht.cols, out


class CAT_HTIHT(nn.Module):

    def __init__(self, vote_index, inplanes, outplanes):
        super(CAT_HTIHT, self).__init__()
        self.htiht = HTIHT(vote_index, inplanes, outplanes)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Conv2d(inplanes+outplanes, inplanes, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        ht, y = self.htiht(x)
        out = self.conv_cat(torch.cat([x,y], dim=1))
        out = self.bn(out)
        out = self.relu(out)

        # 
        # import matplotlib.pyplot as plt
        # for k in range(16):
        #     kk =np.random.randint(low=0, high=128, size=(1)).item()
        #     fig, axs = plt.subplots(nrows=3, ncols=1)
        #     axs = axs.ravel()
        #     ax = axs[0]
        #     ax.imshow(x[0, kk].detach(), cmap='gray')
        #     ax.set_title(x[0, kk].detach().max().item())
        # 
        #     ax = axs[1]
        #     ax.imshow(out[0, kk].detach(), cmap='gray')
        #     ax.set_title(out[0, kk].detach().max().item())
        #     ax = axs[2]
        #     ax.imshow(y[0, kk].detach(), cmap='gray')
        #     ax.set_title(y[0, kk].detach().max().item())
        #     plt.suptitle(str(kk))
        #     plt.show()
        return ht, out
