import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.nn.modules.utils import _pair
import scipy.io as sio
import random
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import time
from models.HT import HT

class HTLoss(torch.nn.Module):
    def __init__(self, device):
        super(HTLoss, self).__init__()
        name = "vote_index_26_122_3_1.mat"
        vote_index_26_122 = sio.loadmat(name)['vote_index']
        vote_index_26_122 = torch.from_numpy(vote_index_26_122).float().contiguous()
        vote_index_26_122.requires_grad=False
        vote_index_26_122 = vote_index_26_122.to(device)
        print('vote_index', vote_index_26_122.shape)

        self.ht = HT(vote_index_26_122)

    def forward(self, ht, pred, pred_exist):
        batch, channel, _, _ = ht.size()
        with torch.no_grad():
            pred_exist = pred_exist.gt(0.9).float().unsqueeze(1)
            pred_exist = pred_exist.expand(batch, channel, 4)
            pred_exist.requires_grad=False
            ### ht for gt:
            pred = F.interpolate(pred, size=(26, 122))
            pred_ht = self.ht(pred)
            idx = pred_ht.view(batch, 4, -1).argmax(2).unsqueeze(1).expand(batch, channel, 4)
            del pred, pred_ht


        ##############visualize the ht loss part ###################
        # print('ht0', F.softmax(ht0.view(batch, channel, -1), dim=2).gather(dim=2, index=idx0))
        # # print('ht0 max', F.softmax(ht0.view(batch, channel, -1), dim=2).max(2)[0])
        # print('ht1', F.softmax(ht1.view(batch, channel, -1), dim=2).gather(dim=2, index=idx0))
        # # print('ht1 max', F.softmax(ht1.view(batch, channel, -1), dim=2).max(2)[0])

        # _, _, h, w =ht1.size()
        # for kk in range(128):
        #     # kk =np.random.randint(low=0, high=128, size=(1)).item()
        #     ht_show = torch.zeros(h, w, 3)
        #     ht_pred = ht1[0, kk]
        #     ht_pred = ht_pred/(ht_pred.max().item()+1e-12)
        #     id = idx1[0, kk]
        #     print('ht_show', ht_show.size(), id.size(),ht_pred.size())
        #     ht_gt= ht_pred.view(-1).index_fill(dim=0, index=id, value=1).view(h,w)
        #
        #     ht_show[:,:,1]=ht_gt
        #     ht_show[:,:,0]=ht_pred
        #
        #     fig, axs = plt.subplots(nrows=1, ncols=2)
        #     axs = axs.ravel()
        #     ax = axs[0]
        #     ax.imshow(ht_show)
        #     ax.set_title(ht_show.detach().max().item())
            # ax = axs[1]
            # ax.imshow(ht_pred.detach(), cmap='gray')
            # ax.set_title(ht_pred.detach().max().item())
            # plt.suptitle(str(kk))
            # plt.show()

        ht_max_log = -1.0 * (F.normalize(ht, dim=2, p=1).view(batch, channel, -1).gather(dim=2, index=idx)+1e-12).log()
        loss = ht_max_log * pred_exist
        loss = loss.mean(2).mean(1)
        return loss


class TotalLoss(torch.nn.Module):
    def __init__(self, ignore_index, weight, device):
        super(TotalLoss, self).__init__()
        self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index, weight=weight, reduction='none')
        self.criterion_exist = torch.nn.BCELoss(reduction='none')

        self.ht_weight = 0.01
        print('self.ht_weight', self.ht_weight)
        self.criterion_ht = HTLoss(device)

    def forward(self, output, target, output_exist, target_exist, ht, semi_flag):
        batch, channel, h, w = output.size()
        loss = self.criterion(F.log_softmax(output, dim=1), target).mean(2).mean(1)  # b
        loss_exist = self.criterion_exist(output_exist, target_exist)  # bx4
        target = target.unsqueeze(1).float()
        target = torch.cat([target.eq(1.0), target.eq(2.0), target.eq(3.0), target.eq(4.0)], dim=1).float()
        # print('output_exist', output_exist.gt(0.9).sum(), target_exist.nelement())
        if semi_flag.sum()==0:
            loss_ht = self.criterion_ht(ht, target, target_exist)  # b
            # loss_tot = loss.mean() + loss_exist.mean() * 0.1 # + loss_ht.mean() * self.ht_weight
            loss_tot = loss.mean() + loss_exist.mean() * 0.1  + loss_ht.mean() * self.ht_weight

        if semi_flag.sum()==len(semi_flag):
            exist_new =output_exist.clone().detach()
            loss_ht = self.criterion_ht(ht, output.softmax(1)[:, 1:5], exist_new)  # b
            loss_tot = loss_ht.mean() * self.ht_weight

        if semi_flag.sum()<len(semi_flag) and semi_flag.sum()>0:

            exist_new =output_exist.clone().detach()
            exist_new[~semi_flag] = target_exist[~semi_flag]
            target[semi_flag] = output.softmax(1)[:, 1:5][semi_flag]
            loss_ht = self.criterion_ht(ht, target, exist_new)  # b
            # exist_new =output_exist.clone().detach()
            # loss_ht = self.criterion_ht(ht, output.softmax(1)[:, 1:5], exist_new)  # b


            ### supervised
            loss_ = loss[~semi_flag]
            loss_exist_ = loss_exist[~semi_flag]

            # ### unsupervised loss
            # loss_ht_ = loss_ht[semi_flag]
            # loss_tot = loss_.mean() + loss_exist_.mean() * 0.1 + loss_ht_.mean() * self.ht_weight

            ### unsupervised loss
            loss_tot = loss_.mean() + loss_exist_.mean() * 0.1 + loss_ht.mean() * self.ht_weight

        return loss.mean(), loss_exist.mean(), loss_ht.mean(), loss_tot

