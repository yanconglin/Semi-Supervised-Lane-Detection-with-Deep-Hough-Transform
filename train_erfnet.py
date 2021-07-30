import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import utils.transforms as tf
import numpy as np
import random
import scipy.io as sio
import models
from loss import TotalLoss
import dataset as ds
from models.HT import hough_transform
from options.options import parser
import multiprocessing as mp

mp.set_start_method('spawn', force=True)
# mp.set_start_method('spawn')
best_mIoU = 0

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.fastest = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    global args, best_mIoU, device
    args = parser.parse_args()
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
        args.gpus = len(args.gpus)
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        device_name = "cpu"
        print("CUDA is not available")
    device = torch.device(device_name)

    if args.dataset == 'CULane':
        num_class = 5
        ignore_label = 255
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    ### pre-compute the vote_index. Hard_coded there
    if not os.path.exists("vote_index_26_122_3_1.mat"):
        vote_index = hough_transform(26, 122, 3, 1)
        sio.savemat('vote_index_26_122_3_1.mat', {'vote_index': vote_index.numpy()})
        print('vote_index', vote_index.shape, type(vote_index), vote_index.dtype)
        print('vote_index  memory MB', vote_index.size(), vote_index.element_size() * vote_index.nelement() / (1024 * 1024))

    set_seed(0)
    model = models.ERFNet_HT(num_classes=num_class, device=device)
    input_mean = model.input_mean
    input_std = model.input_std
    model = torch.nn.DataParallel(model).to(device)
    print('model', model)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num of total parameters', total_params)
    print('num of trainable parameters', train_params)

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        ckpt_name = []
        cnt = 0
        for name, param in state_dict.items():
            if name not in list(own_state.keys()) or 'output_conv' in name:
                 ckpt_name.append(name)
                 continue
            own_state[name].copy_(param)
            cnt += 1
        print('#reused param: {}'.format(cnt))
        return model

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume,  map_location=lambda storage, loc: storage)
            args.start_epoch = checkpoint['epoch']
            # model = load_my_state_dict(model, checkpoint['state_dict'])
            torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            print(("=> loaded checkpoint (epoch {})".format(checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True
    cudnn.fastest = True

    # Data loading code
    set_seed(0)
    train_loader = torch.utils.data.DataLoader(
        getattr(ds, args.dataset.replace("CULane", "VOCAug") + 'DataSet')(data_list=args.train_list, transform=torchvision.transforms.Compose([
            tf.GroupRandomScale(size=(0.595, 0.621), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            tf.GroupRandomCropRatio(size=(args.img_width, args.img_height)),
            tf.GroupRandomRotation(degree=(-1, 1), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST), padding=(input_mean, (ignore_label, ))),
            tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ])), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False, drop_last=True)

    set_seed(0)
    val_loader = torch.utils.data.DataLoader(
        getattr(ds, args.dataset.replace("CULane", "VOCAug") + 'DataSet')(data_list=args.val_list, transform=torchvision.transforms.Compose([
            tf.GroupRandomScale(size=(0.595, 0.621), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            tf.GroupRandomCropRatio(size=(args.img_width, args.img_height)),
            tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ]), mode='valid'), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    print('train_loader val_loader size', len(train_loader), len(val_loader))
    print('args',args)
    # define loss function (criterion) optimizer and evaluator
    weights = [1.0 for _ in range(5)]
    weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = TotalLoss(ignore_index=ignore_label, weight=class_weights, device=device).to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    evaluator = EvalSegmentation(num_class, ignore_label)

    for epoch in range(args.epochs):  # args.start_epoch
        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        # if epoch%2==0:
        #     adjust_learning_rate(optimizer, epoch//2, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            mIoU = validate(val_loader, model, criterion, evaluator)
            # remember best mIoU and save checkpoint
            is_best = mIoU > best_mIoU
            best_mIoU = max(mIoU, best_mIoU)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_mIoU': best_mIoU,
            }, is_best)



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_exist = AverageMeter()
    losses_ht = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_exist, semi_flag) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)
        target_exist = target_exist.to(device)
        semi_flag = semi_flag.to(device)

        # compute output
        output, output_exist, ht = model(input)
        # print("output", output.size(), output_exist.size(), output_ht.size())
        loss, loss_exist, loss_ht, loss_tot= criterion(output, target, output_exist, target_exist, ht, semi_flag)

        # compute gradient and do SGD step
        if loss_tot is None: continue
        optimizer.zero_grad()
        loss_tot.backward()
        optimizer.step()

        # print('semi_flag', semi_flag, target_exist)
        # print('loss, loss_exist, loss_ht', loss, loss_exist, loss_ht)
        # measure accuracy and record loss
        losses.update(loss.data.item(), input.size(0))
        losses_exist.update(loss_exist.data.item(), input.size(0))
        losses_ht.update(loss_ht.data.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print((
                'Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Loss_ht {loss_ht.val:.8f} ({loss_ht.avg:.8f})\t'
                'Loss_exist {loss_exist.val:.4f} ({loss_exist.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                    loss_ht=losses_ht,
                    loss_exist=losses_exist, lr=optimizer.param_groups[-1]['lr'])))
            batch_time.reset()
            data_time.reset()
            losses.reset()


def validate(val_loader, model, criterion, evaluator):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_exist = AverageMeter()
    losses_ht = AverageMeter()
    IoU = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target, target_exist, semi_flag) in enumerate(val_loader):
            # print('size', input.size(), target.size())
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(nrows=1, ncols=2)
            # axs = axs.ravel()
            # ax = axs[0]
            # ax.imshow(input[0,0]/255.)
            # ax = axs[1]
            # ax.imshow(target[0])
            # plt.show()
            input = input.to(device)
            target = target.to(device)
            target_exist = target_exist.to(device)
            semi_flag = semi_flag.to(device)

            # compute output
            output, output_exist, ht = model(input)
            loss, loss_exist, loss_ht, _ = criterion(output, target, output_exist, target_exist, ht, semi_flag)
            # measure accuracy and record loss
            pred = output.data.cpu().numpy().transpose(0, 2, 3, 1)
            pred = np.argmax(pred, axis=3).astype(np.uint8)
            IoU.update(evaluator(pred, target.cpu().numpy()))
            losses.update(loss.data.item(), input.size(0))
            losses_exist.update(loss_exist.data.item(), input.size(0))
            losses_ht.update(loss_ht.data.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % args.print_freq == 0:
                acc = np.sum(np.diag(IoU.sum)) / float(np.sum(IoU.sum))
                mIoU = np.diag(IoU.sum) / (1e-20 + IoU.sum.sum(1) + IoU.sum.sum(0) - np.diag(IoU.sum))
                mIoU = np.sum(mIoU) / len(mIoU)
                print(('Test: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Loss_exist {loss_exist.val:.4f} ({loss_exist.avg:.4f})\t'
                       'loss_ht {loss_ht.val:.4f} ({loss_ht.avg:.4f})\t'
                       'Pixels Acc {acc:.3f}\t'
                       'mIoU {mIoU:.3f}'.format(
                    i, len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    loss_exist=losses_exist,
                    loss_ht=losses_ht,
                    acc=acc,
                    mIoU=mIoU)))

    acc = np.sum(np.diag(IoU.sum)) / float(np.sum(IoU.sum))
    mIoU = np.diag(IoU.sum) / (1e-20 + IoU.sum.sum(1) + IoU.sum.sum(0) - np.diag(IoU.sum))
    mIoU = np.sum(mIoU) / len(mIoU)
    print(('Testing Results: Pixels Acc {acc:.3f}\tmIoU {mIoU:.3f} ({bestmIoU:.4f})\tLoss {loss.avg:.5f}'.format(
        acc=acc, mIoU=mIoU, bestmIoU=max(mIoU, best_mIoU), loss=losses)))

    return mIoU


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.exists('trained'):
        os.makedirs('trained')

    filename = os.path.join('trained', str(state["epoch"])+'_'.join((args.snapshot_pref, args.method.lower(), filename)))
    torch.save(state, filename)
    if is_best:
        best_name = os.path.join('trained', '_'.join((args.snapshot_pref, args.method.lower(), 'model_best.pth.tar')))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = (gt != self.ignore_label)
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength=self.num_class ** 2).reshape(self.num_class, self.num_class)
        return hs


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    decay = ((1 - float(epoch) / args.epochs)**(0.9))
    lr = args.lr * decay
    decay = args.weight_decay
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
        param_group['weight_decay'] = decay


if __name__ == '__main__':
    main()
