# ERFNET full network definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.HT import CAT_HTIHT, hough_transform
import scipy.io as sio

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64))    # 0-1
        for x in range(0, 5):  # 5 times, 1-6
            self.layers.append(non_bottleneck_1d(64, 0.1, 1))
        self.layers.append(DownsamplerBlock(64, 128)) # 6-7
        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))

        # only for encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

        self.layers_1 = nn.Sequential(*self.layers[0:6])
        self.layers_2 = nn.Sequential(*self.layers[6:])
        # self.ht_layers = nn.ModuleList()
        # self.ht_layers.append(CAT_HTIHT(vote_index["52_244"], inplanes=64, outplanes=16))
        # self.ht_layers.append(CAT_HTIHT(vote_index["26_122"], inplanes=128, outplanes=16))

    def forward(self, input, predict=False):

        output = self.initial_block(input)
        output = self.layers_1(output)
        # output = self.ht_layers[0](output)
        output = self.layers_2(output)
        # output = self.ht_layers[1](output)

        if predict:
            output = self.output_conv(output)
        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        # print('Decoder in', input.size())

        for layer in self.layers:
            output = layer(output)
            # print('Decoder', output.size())
        output = self.output_conv(output)
        # print('Decoder final', output.size())
        return output


class Lane_exist(nn.Module):
    def __init__(self, num_output=4):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(128, 32, (3, 3), stride=1, padding=(4, 4), bias=False, dilation=(4, 4)))
        self.layers.append(nn.BatchNorm2d(32, eps=1e-03))

        self.layers_final = nn.ModuleList()

        self.layers_final.append(nn.Dropout2d(0.1))
        self.layers_final.append(nn.Conv2d(32, 5, (1, 1), stride=1, padding=(0, 0), bias=True))

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear1 = nn.Linear(3965, 128)
        self.linear2 = nn.Linear(128, num_output)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = F.relu(output)

        for layer in self.layers_final:
            output = layer(output)

        output = F.softmax(output, dim=1)
        output = self.maxpool(output)
        # print(output.shape)
        output = output.view(-1, 3965)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        return output.sigmoid()


class ERFNet_HT(nn.Module):
    def __init__(self, num_classes, device=None):
        super().__init__()
        name = "vote_index_26_122_3_1.mat"
        vote_index_26_122 = sio.loadmat(name)['vote_index']
        vote_index_26_122 = torch.from_numpy(vote_index_26_122).float().contiguous()
        vote_index_26_122.requires_grad=False
        vote_index_26_122 = vote_index_26_122.to(device)
        print('vote_index', vote_index_26_122.shape)

        self.encoder = Encoder(num_classes)
        self.decoder = Decoder(num_classes)
        self.lane_exist = Lane_exist(4)  # num_output
        self.input_mean = [103.939, 116.779, 123.68]  # [0, 0, 0]
        self.input_std = [1, 1, 1]

        self.ht=CAT_HTIHT(vote_index_26_122, inplanes=128, outplanes=16)

    def forward(self, input):

        output = self.encoder(input)
        ht, output = self.ht(output)
        return self.decoder(output), self.lane_exist(output), ht

