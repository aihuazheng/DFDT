from torch.autograd import Variable
import torch.nn.functional as F
import time
import torch
import numpy as np
#from sklearn import metrics
from utils.Logger import AverageMeter
from main_rap.evaluate import mA, example_based, mA_LB


def testing(data_loader, model, opt):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()

    output_buffer = [[],[],[],[],[],[]]
    gt_buffer = []


    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            if not opt.no_cuda:
                inputs = inputs.cuda()

        for m in range(6):

            inputs = F.interpolate(inputs, size=opt.scale_size[m], mode="nearest")
            inputs = Variable(inputs, volatile=True)
            outputs = model(inputs)
            outputs = F.sigmoid(outputs)

            output_buffer[m] += list(outputs.data.cpu().numpy())

        gt_buffer += list(targets.data.cpu().numpy())

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
            i + 1, len(data_loader), batch_time=batch_time, data_time=data_time))

    ######################################################
    gt_results = np.stack(gt_buffer, axis=0)
    gt = np.where(gt_results > 0.5, 1, 0)[:, 0:51]

    results_ensemble = np.zeros(gt_results.shape)

    for m in range(6):
        output_results = np.stack(output_buffer[m], axis=0)
        if m!=3:
            results_ensemble += output_results
        attr = np.where(output_results > 0.5, 1, 0)[:, 0:51]

        ####################################################

        print mA(attr, gt)
        print example_based(attr, gt)

        print "-------------------------------"
        print mA_LB(attr, gt)
        print ('####################################')


    attr = np.where(results_ensemble > 2, 1, 0)[:, 0:51]
    print mA(attr, gt)
    print example_based(attr, gt)

    print "-------------------------------"
    print mA_LB(attr, gt)
    print ('####################################')


    # print ('per-class based metrics')
    # print ('acc        pre        recal        f1')
    # # mAP = metrics.average_precision_score(gt.T, attr.T)
    # # print 'mAP score:', mAP
    # print mA(attr.T, gt.T)
    # print example_based(attr.T, gt.T)
    #
    # print ('micro based metrics')
    # print ('acc        pre        recal        f1')
    # # mAP = metrics.average_precision_score(attr.reshape((1, -1)), gt.reshape((1, -1)))
    # # print 'mAP score:', mAP
    # print mA(attr.reshape((-1, 1)), gt.reshape((-1, 1)))
    # print example_based(attr.reshape((1, -1)), gt.reshape((1, -1)))

    # attr, gt = attr.reshape((-1, 1)), gt.reshape((-1, 1))
    # print metrics.accuracy_score(gt, attr)
    # print metrics.precision_score(gt, attr)
    # print metrics.recall_score(gt, attr)
    # print metrics.f1_score(gt, attr)
