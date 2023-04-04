import torch
import time
import os
import random
from utils.Logger import AverageMeter
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
#
def adjust_learning_rate(optimizer, iter, opt):

    if iter % 10 == 0 and iter < opt.max_iter:
        new_lr = opt.learning_rate*(1 - float(iter) / opt.max_iter) ** opt.power
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        # optimizer.param_groups[1]['lr'] = 10*new_lr

# def adjust_learning_rate(optimizer, epoch, opt):
#     """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""
#     if (epoch - 1) % opt.lr_steps == 0 and epoch != 1:
#         print("Current learning rate is decaying.")
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 0.1 * param_group['lr']


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # adjust_learning_rate(optimizer, epoch, opt)

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            inputs = inputs.cuda()
            targets = targets.float().cuda()

        # print targets
        #print inputs
        inputs = torch.autograd.Variable(inputs, requires_grad=True)
        targets = torch.autograd.Variable(targets, requires_grad=False)

        ##########################################
        random.seed()
        scale_size = opt.scale_size[random.randint(0,5)]
        # print scale_size


        inputs = F.interpolate(inputs, size=scale_size, mode="nearest")
        ##########################################
        outputs = model(inputs)
        loss = criterion(outputs, targets) / opt.batch_size

        losses.update(loss.item(), inputs.size(0))

        # acc = calculate_accuracy(outputs, targets)

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
            'loss': losses.val,
            'lr': optimizer.param_groups[len(optimizer.param_groups) - 1]['lr']
        })
        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i + 1, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'lr': optimizer.param_groups[len(optimizer.param_groups) - 2]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.pretrain_path, 'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)