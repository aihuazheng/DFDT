import argparse
import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
from torch import optim
from main_rap.Dataset import AttributeDataset
from main_rap.spatial_transforms import (Compose, Normalize, Scale, RandomHorizontalFlip, ToTensor)
from test_baseline import testing
from train import train_epoch
from utils.Logger import Logger
from model_ms import MultiScaleClassifier


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', default='../log/RAP/multiscale', type=str,
                        help='Result directory path')
    parser.add_argument('--eps', default=0.01, type=float)
    parser.add_argument('--sample_size', default=[384, 384], type=int, help='Height and width of inputs')
    parser.add_argument('--scale_size', default=[[384, 192],
                                                 [320, 240], [320, 160],
                                                 [256, 256], [256, 192], [256, 128],
                                                 ], type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--power', default=0.9, type=float, help='Power')
    parser.add_argument('--max_iter', default=1040*90, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight Decay')
    parser.add_argument('--mean', default=[0.485, 0.456, 0.406], type=float, help='Weight Decay')
    parser.add_argument('--std', default=[0.229, 0.224, 0.225], type=float, help='Weight Decay')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=90, type=int, help='Number of total epochs to run')
    parser.add_argument('--print_freq', default=50, type=int, help='print')
    parser.add_argument('--begin_epoch', default=1, type=int,
                        help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--resume_path', default='../saved_model/RAP/multiscale/save_90.pth', type=str,
                        help='Save data (.pth) of previous training')
    parser.add_argument('--pretrain_path', default='../saved_model/RAP/multiscale', type=str,
                        help='Pretrained model (.pth)')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=True)
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=True)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--n_threads', default=8, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint', default=90, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--norm_value', default=255, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--model', default='resnet', type=str,
                        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth', default=1, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--lr_steps', default=20, type=int)
    parser.add_argument('--dataset', default='RAP', type=str, help='( RAP | PETA | PA100k )')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    torch.cuda.manual_seed(123)

    opts = parse_opts()
    opts.arch = '{}-{}'.format(opts.model, opts.model_depth)
    print (opts)
    cudnn.benchmark = True

    ##################################################################
    ##################################################################

    if not False:
        spatial_transform = Compose([Scale(opts.sample_size),
                                     RandomHorizontalFlip(),
                                     ToTensor(opts.norm_value),
                                     Normalize(opts.mean, opts.std)])

        training_data = AttributeDataset(datasetID=opts.dataset,
                                         split='train',
                                         splitID=0,
                                         spatial_transform=spatial_transform)

        train_loader = torch.utils.data.DataLoader(training_data, batch_size=opts.batch_size,
                                                   shuffle=True, num_workers=opts.n_threads, pin_memory=True)

        train_logger = Logger(os.path.join(opts.result_path, 'train.log'),
                              ['epoch', 'loss', 'lr'])
        train_batch_logger = Logger(os.path.join(opts.result_path, 'train_batch.log'),
                                    ['epoch', 'batch', 'iter', 'loss', 'lr'])

    if opts.nesterov:
        dampening = 0
    else:
        dampening = opts.momentum

    #################################################################
    ##################################################################
    if not opts.no_val:
        spatial_transform = Compose([Scale(opts.sample_size),
                                     RandomHorizontalFlip(),
                                     ToTensor(opts.norm_value),
                                     Normalize(opts.mean, opts.std)])

        validation_data = AttributeDataset(datasetID=opts.dataset,
                                           split='val',
                                           splitID=None,
                                           spatial_transform=spatial_transform)

        val_loader = torch.utils.data.DataLoader(validation_data, batch_size=opts.batch_size,
                                                 shuffle=False, num_workers=opts.n_threads, pin_memory=True)

        val_logger = Logger(os.path.join(opts.result_path, 'val.log'),
                            ['epoch', 'loss'])

    ##################################################################

    model = MultiScaleClassifier(num_classes=84)
    model = model.cuda()
    print ("Building model ... ")

    cudnn.benchmark = True
    params = model.parameters()

    criterion = nn.BCEWithLogitsLoss(size_average=False)
    if not opts.no_cuda:
        criterion = criterion.cuda()

    optimizer = optim.SGD(params,lr=opts.learning_rate,
                          momentum=opts.momentum, dampening=dampening,
                          weight_decay=opts.weight_decay, nesterov=opts.nesterov)

    if opts.resume_path:
        print('loading checkpoint {}'.format(opts.resume_path))
        checkpoint = torch.load(opts.resume_path)
        assert opts.arch == checkpoint['arch']

        opts.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opts.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])


    print('run')
    for i in range(opts.begin_epoch, opts.n_epochs + 1):
        if not opts.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opts, train_logger, train_batch_logger)

        if not opts.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opts, val_logger)

        # if not opts.no_train and not opts.no_val:
        #     scheduler.step(validation_loss)

        ##################################################################
        ##################################################################

    if opts.test:
        spatial_transform = Compose([Scale(opts.sample_size),
                                     ToTensor(opts.norm_value),
                                     Normalize(opts.mean, opts.std)])

        test_data = AttributeDataset(datasetID=opts.dataset,
                                     split='test',
                                     splitID=0,
                                     spatial_transform=spatial_transform)

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=opts.batch_size,
                                                  shuffle=False, num_workers=opts.n_threads, pin_memory=True)

        feats_query = testing(test_loader, model, opts)