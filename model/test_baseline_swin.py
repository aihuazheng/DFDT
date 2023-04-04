from torch.autograd import Variable
import torch.nn.functional as F
import time
import torch
import numpy as np
#from sklearn import metrics
from utils.Logger import AverageMeter
from utils.evaluate import mA, example_based, mA_LB,mA1
from utils.metrics import scores



# select_attr = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 1, 2, 3, 0, 4, 5, 6, 7, 8, 43, 44, 45, 46, 47, 48, 49, 50]
def testing_RAP_swin(data_loader, model, opt):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = [[],[],[],[],[],[]]
    gt_buffer = []


    for i, (imgs, gt_label, imgname) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            if not opt.no_cuda:
                imgs = imgs.cuda()
                gt_label = gt_label.float().cuda()

        for m in range(6):
            inputs = torch.autograd.Variable(imgs)
            targets = torch.autograd.Variable(gt_label)
            output = model(inputs)
            output = F.sigmoid(output)
            output_buffer[m] += list(output.data.cpu().numpy())


        gt_buffer += list(targets.data.cpu().numpy())

        batch_time.update(time.time() - end_time)
        end_time = time.time()

    ######################################################
    gt_results = np.stack(gt_buffer, axis=0)
    gt_results = np.where(gt_results > 0.5, 1, 0)

    ######################################
    results_ensemble = np.zeros(gt_results.shape)

    for m in range(6):
        output_results = np.stack(output_buffer[m], axis=0)
        if m!=0:
            results_ensemble += output_results
        attr = np.where(output_results > 0.5, 1, 0)

        ####################################################

        print (mA(attr, gt_results))
        print (example_based(attr, gt_results))

        print ("-------------------------------")
        print (mA_LB(attr, gt_results))
        print ('####################################')


    attr = np.where(results_ensemble > 2, 1, 0)
    print (mA(attr, gt_results))
    print (example_based(attr, gt_results))

    print ("-------------------------------")
    print (mA_LB(attr, gt_results))
    print ('####################################')




# selected_attr = [99, 100, 101, 102, 48, 49, 59, 21, 60, 22, 3, 23, 62, 82, 24, 5, 104, 54, 6, 7, 55, 27, 56, 83, 84, 66, 28, 67, 85, 29, 9, 69, 33, 34, 35]
# selected_attr = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5, 17, 20, 22, 0, 1, 2, 3, 16]
def testing_PETA_swin(data_loader, model, opt):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = [[],[],[],[],[],[]]
    gt_buffer = []


    for i, (imgs, gt_label, imgname) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            if not opt.no_cuda:
                inputs = imgs.cuda()
                targets = gt_label.float().cuda()

        for m in range(6):
            inputs = torch.autograd.Variable(inputs)
            targets = torch.autograd.Variable(targets)

            output = model(inputs)
            output = F.sigmoid(output)
            #print(output)
            #output_parsing = np.argmax(output_parsing, axis=1)
            output_buffer[m] += list(output.data.cpu().numpy())

        # for j in range(output.size(0)):
        #     # output_buffer.append(output[j].data.cpu())
        #     gt_buffer.append(targets[j].data.cpu())
        gt_buffer += list(targets.data.cpu().numpy())
        # output_buffer_parsing  += list(np.argmax(output_parsing.data.cpu().numpy(), axis=1))
        # gt_buffer_parsing += list(label_parsing.data.cpu().numpy())

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # print('[{}/{}]\t'
        #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
        #     i + 1, len(data_loader), batch_time=batch_time, data_time=data_time))

    ######################################################
    # output_results = torch.stack(output_buffer)
    #gt_results = torch.stack(gt_buffer)
    gt_results = np.stack(gt_buffer, axis=0)
    # output_results = np.squeeze(output_results.numpy()).astype(np.float32)
    #gt_results = np.squeeze(gt_results.numpy()).astype(np.float32)
    gt_results = np.where(gt_results > 0.5, 1, 0)
    ######################################
    results_ensemble = np.zeros(gt_results.shape)

    for m in range(6):
        output_results = np.stack(output_buffer[m], axis=0)
        if m!=0:
            results_ensemble += output_results
        attr = np.where(output_results > 0.5, 1, 0)

        ####################################################

        print (mA(attr, gt_results))
        print (example_based(attr, gt_results))

        print ("-------------------------------")
        print (mA_LB(attr, gt_results))
        print ('####################################')


    attr = np.where(results_ensemble > 2, 1, 0)
    print (mA(attr, gt_results))
    # print (mA1(att, gt_results[:,selected_attr]))
    print (example_based(attr, gt_results))

    print ("-------------------------------")
    print (mA_LB(attr, gt_results))
    print ('####################################')

def testing_PA100K_swin(data_loader, model, opt):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = [[],[],[],[],[],[]]
    gt_buffer = []


    for i, (imgs, gt_label, imgname) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            if not opt.no_cuda:
                inputs = imgs.cuda()
                targets = gt_label.float().cuda()

        for m in range(6):
            inputs = torch.autograd.Variable(inputs)
            targets = torch.autograd.Variable(targets)

            output = model(inputs)
            output = torch.sigmoid(output)
            #output_parsing = np.argmax(output_parsing, axis=1)
            output_buffer[m] += list(output.data.cpu().numpy())

        # for j in range(output.size(0)):
        #     # output_buffer.append(output[j].data.cpu())
        #     gt_buffer.append(targets[j].data.cpu())
        gt_buffer += list(targets.data.cpu().numpy())
        # output_buffer_parsing  += list(np.argmax(output_parsing.data.cpu().numpy(), axis=1))
        # gt_buffer_parsing += list(label_parsing.data.cpu().numpy())

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # print('[{}/{}]\t'
        #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
        #     i + 1, len(data_loader), batch_time=batch_time, data_time=data_time))

    ######################################################
    # output_results = torch.stack(output_buffer)
    # gt_results = torch.stack(gt_buffer)
    gt_results = np.stack(gt_buffer, axis=0)
    # output_results = np.squeeze(output_results.numpy()).astype(np.float32)
    #gt_results = np.squeeze(gt_results.numpy()).astype(np.float32)
    gt_results = np.where(gt_results > 0.5, 1, 0)
    # output_results = np.squeeze(output_results.numpy()).astype(np.float32)
    # gt_results = np.squeeze(gt_results.numpy()).astype(np.float32)

    ######################################
    results_ensemble = np.zeros(gt_results.shape)

    for m in range(6):
        output_results = np.stack(output_buffer[m], axis=0)
        if m!=0:
            results_ensemble += output_results
        attr = np.where(output_results > 0.5, 1, 0)

        ####################################################

        print (mA(attr, gt_results))
        print (example_based(attr, gt_results))

        print ("-------------------------------")
        print (mA_LB(attr, gt_results))
        print ('####################################')


    attr = np.where(results_ensemble > 2, 1, 0)
    print (mA(attr, gt_results))
    print (example_based(attr, gt_results))

    print ("-------------------------------")
    print (mA_LB(attr, gt_results))
    print ('####################################')
