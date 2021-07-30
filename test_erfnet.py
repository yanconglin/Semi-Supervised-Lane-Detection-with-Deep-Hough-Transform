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
import models
from loss import TotalLoss
# from models import sync_bn
import dataset as ds
from options.options import parser
import torch.nn.functional as F
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
# mp.set_start_method('spawn')

best_mIoU = 0


def main():
    global args, best_mIoU, device
    args = parser.parse_args()
    args = parser.parse_args()
    device_name = "cpu"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    if args.dataset == 'VOCAug' or args.dataset == 'VOC2012' or args.dataset == 'COCO':
        num_class = 21
        ignore_label = 255
        scale_series = [10, 20, 30, 60]
    elif args.dataset == 'Cityscapes':
        num_class = 19
        ignore_label = 255
        scale_series = [15, 30, 45, 90]
    elif args.dataset == 'ApolloScape':
        num_class = 37
        ignore_label = 255
    elif args.dataset == 'CULane':
        num_class = 5
        ignore_label = 255
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model = models.ERFNet_HT(num_classes=num_class, device=device)
    input_mean = model.input_mean
    input_std = model.input_std
    # policies = model.get_optim_policies()
    # model = torch.nn.DataParallel(model, device_ids=range(args.gpus)).to(device)
    model = torch.nn.DataParallel(model).to(device)
    # print('model', model)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num of total parameters', total_params)
    print('num of trainable parameters', train_params)
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            args.start_epoch = checkpoint['epoch']
            best_mIoU = checkpoint['best_mIoU']
            torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            # model.load_state_dict(torch.load(checkpoint['state_dict'], map_location=lambda storage, loc: storage))
            print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))


    cudnn.benchmark = True
    cudnn.fastest = True

    # Data loading code

    test_loader = torch.utils.data.DataLoader(
        getattr(ds, args.dataset.replace("CULane", "VOCAug") + 'DataSet')(data_list=args.val_list, transform=torchvision.transforms.Compose([
            tf.GroupRandomScaleNew(size=(args.img_width, args.img_height), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ]), mode='test'), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
    print('test_loader size', len(test_loader))
    # define loss function (criterion) optimizer and evaluator
    weights = [1.0 for _ in range(5)]
    weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = TotalLoss(ignore_index=ignore_label, weight=class_weights, device=device).to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    evaluator = EvalSegmentation(num_class, ignore_label)
    print('args', args)
    ### evaluate ###
    with torch.no_grad():
        validate(test_loader, model, criterion, 0, evaluator)
    return


def validate(val_loader, model, criterion, iter, evaluator, logger=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()
    mIoU = 0
    num_correct = 0
    num_total = 0
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, target_exist, img_name) in enumerate(val_loader):
        print('batch', i)
        # compute output
        output, output_exist, ht = model(input)

        # measure accuracy and record loss

        output = F.softmax(output, dim=1)

        num_correct += (output_exist.data.cpu().gt(0.5).float()==target_exist.float()).sum().item()
        num_total += target_exist.nelement()
        print('accuracy', num_correct/num_total, num_correct, num_total)

        pred = output.data.cpu().numpy() # BxCxHxW
        pred_exist = output_exist.data.cpu().numpy() # BxO

        for cnt in range(len(img_name)):
            print('directory', img_name[cnt])
            directory = 'predicts/vgg_SCNN_DULR_w9' + img_name[cnt][:-10]
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_exist = open('predicts/vgg_SCNN_DULR_w9'+img_name[cnt].replace('.jpg', '.exist.txt'), 'w')
            for num in range(4):
                prob_map = (pred[cnt][num+1]*255).astype(int)
                save_img = cv2.blur(prob_map,(9,9))
                cv2.imwrite('predicts/vgg_SCNN_DULR_w9'+img_name[cnt].replace('.jpg', '_'+str(num+1)+'_avg.png'), save_img)
                if pred_exist[cnt][num] > 0.5:
                    file_exist.write('1 ')
                else:
                    file_exist.write('0 ')
            file_exist.close()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time)))

    print('finished, #test:{}'.format(i) )

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
        hs = np.bincount(sumim[locs], minlength=self.num_class**2).reshape(self.num_class, self.num_class)
        return hs


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    decay = ((1 - float(epoch) / args.epochs)**(0.9))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


if __name__ == '__main__':
    main()
