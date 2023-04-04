import torch
import time
import datetime
import os
import random
import torch
import torch.nn.functional as F
from utils.Logger import AverageMeter
import torch.nn as nn

#
def adjust_learning_rate(optimizer, iter, opt):

    if iter % 5 == 0 and iter < opt.max_iter:
        new_lr = opt.learning_rate*(1 - float(iter) / opt.max_iter) ** opt.power
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


# def adjust_learning_rate(optimizer, epoch, opt):
#     """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""
#     if (epoch - 1) % opt.lr_steps == 0 and epoch != 1:
#         print("Current learning rate is decaying.")
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 0.1 * param_group['lr']


def train_epoch_supression_layers2(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_BCE = AverageMeter()
    losses_BCE_W = AverageMeter()
    losses_BCE_s_3_W = AverageMeter()
    losses_BCE_s_2_W = AverageMeter()
    total_loss = AverageMeter()

    end_time = time.time()
    for i, (imgs, gt_label, imgname) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        imgs = imgs.cuda()
        gt_label = gt_label.float().cuda()

        inputs = torch.autograd.Variable(imgs)
        targets = torch.autograd.Variable(gt_label)

        ##########################################
        ##########################################
        random.seed()

        output,output_s_3,output_s_2 = model(inputs)
        ##############    bce
        loss_1 = nn.BCEWithLogitsLoss(size_average=False)(output, targets)
        loss_1 = loss_1/opt.batch_size

        ###############  w bce
        loss_BCE_W,loss_BCE= criterion(output, targets)
        loss_BCE_s_3_W,loss_BCE_s_3 = criterion(output_s_3, targets)
        loss_BCE_s_2_W,loss_BCE_s_2 = criterion(output_s_2, targets)
        # loss_BCE_W = loss_BCE_W/opt.batch_size
        # loss_BCE_s_3_W = loss_BCE_s_3_W/opt.batch_size
        # loss_BCE_s_2_W = loss_BCE_s_2_W/opt.batch_size

        ################  total 
        loss = loss_1 +loss_BCE_s_3_W+loss_BCE_s_2_W

        ################   update
        losses_BCE.update(loss_1.item(), inputs.size(0))
        # losses_BCE_W.update(loss_BCE_W.item(), inputs.size(0))
        losses_BCE_s_3_W.update(loss_BCE_s_3_W.item(), inputs.size(0))
        losses_BCE_s_2_W.update(loss_BCE_s_2_W.item(), inputs.size(0))
        total_loss.update(loss.item(), 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        #
        iter = (epoch - 1) * len(data_loader) + (i + 1)
        adjust_learning_rate(optimizer, iter, opt)


        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': total_loss.val,
            'lr': optimizer.param_groups[len(optimizer.param_groups) - 1]['lr']
        })
        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Total_loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_BCE {loss_BCE.val:.4f} ({loss_BCE.avg:.4f})\t'
                  'Loss_BCE_s_3 {loss_BCE_S_3.val:.4f} ({loss_BCE_S_3.avg:.4f})\t'
                  'Loss_BCE_s_2 {loss_BCE_S_2.val:.4f} ({loss_BCE_S_2.avg:.4f})\t'
                .format(
                epoch, i + 1, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=total_loss, loss_BCE=losses_BCE, loss_BCE_S_3=losses_BCE_s_3_W,loss_BCE_S_2=losses_BCE_s_2_W))

    epoch_logger.log({
        'epoch': epoch,
        'loss': total_loss.avg,
        'lr': optimizer.param_groups[len(optimizer.param_groups) - 2]['lr']
    })
    #if not os.path.exists(opt.pretrain_path):
        #os.mkdir(opt.pretrain_path)
    if epoch % 1 == 0:
        save_file_path = os.path.join(opt.pretrain_path, 'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)