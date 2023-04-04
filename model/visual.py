from ast import Not
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
print("#####"+str(curPath))
root_path = os.path.abspath(os.path.dirname(curPath) + os.path.sep + ".")
sys.path.append(root_path)
   
import argparse
import cv2
import numpy as np
import torch
import timm

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image

from model.visual_dataset import AttributeDataset
from model.swin_IB import Swin_IB
from model.swinT import SwinT
from model.swin_VSD import Swin_VSD
from model.swin_supression import Swin_Supression
from model.swin_supression_layers2_CVPR_join import Swin_Supression_Layers2_CVPR_join
from model.train_baseline_swin import train_epoch_swin
from model.train_baseline_swin_IB import train_epoch_IB
from model.train_baseline_swin_VSD import train_epoch_VSD
from model.train_baseline_swin_supression import train_epoch_supression
from model.test_baseline_swin import testing_PA100K_swin
from model.test_baseline_swin_IB import testing_PA100K_IB
from model.test_baseline_swin_VSD import testing_PA100K_VSD
from model.test_baseline_swin_supression import testing_PA100K_supression


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', default='/data/huimin.wang/log/IB/PA100K/swin_drop/', type=str,
                        help='Result directory path')
    parser.add_argument('--pretrain_path', default='/data/huimin.wang/pre/IB/PA100K/swin_drop/', type=str,
                        help='Pretrained model (.pth)')
    parser.add_argument('--resume_path', default="/data/huimin.wang/pre/IB/PA100K/swin_2/save_31.pth", type=str,
                        help='Save data (.pth) of previous training')
    parser.add_argument('--visual_path', default="/data/huimin.wang/dataset/visual/", type=str,
                        help='visual img (.jpg)')

    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--power', default=0.9, type=float, help='Power')
    parser.add_argument('--max_iter', default=2500*40, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight Decay')
    parser.add_argument('--mean', default=[0.485, 0.456, 0.406], type=float, help='Weight Decay')
    parser.add_argument('--std', default=[0.229, 0.224, 0.225], type=float, help='Weight Decay')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')

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
    parser.add_argument('--checkpoint', default=1, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--norm_value', default=255, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--model', default='resnet', type=str,
                        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth', default=1, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--lr_steps', default=20, type=int)
    parser.add_argument('--dataset', default='PA100k', type=str, help='( RAP | PETA | PA100k )')
    parser.add_argument('--temperature', type=int, default=1, help="the temperature used in knowledge distillation")
    # hyper-parameters
    parser.add_argument('-BCE_loss', type=int, default=1, help="weight of cross entropy loss")
    parser.add_argument('-VSD_loss', type=float, default=2, help="weight of VSD")
    parser.add_argument('-temperature', type=int, default=1, help="the temperature used in knowledge distillation")

    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path',type=str,default="/data/huimin.wang/dataset/5.png",
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth',action='store_true',help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--methods',type=str,default='scorecam',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.methods not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")


    model = SwinT()
    if args.resume_path:
        print('loading checkpoint {}'.format(args.resume_path))
        checkpoint = torch.load(args.resume_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.layers[-1].blocks[-1].norm2]

    if args.methods not in methods:
        raise Exception(f"Method {args.methods} not implemented")

    cam = methods[args.methods](model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)

    # 读取数据
    test_data = AttributeDataset(datasetID=args.dataset,opt=args,split='test',splitID=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,shuffle=False, num_workers=args.n_threads, pin_memory=True)
    
    for i, (rgb_img, targets, file_name) in enumerate(test_loader):

        rgb_img = rgb_img.squeeze(0).detach().numpy()
        print(rgb_img.shape)

        # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
        # rgb_img = cv2.resize(rgb_img, (224, 224))
        # rgb_img = np.float32(rgb_img) / 255

        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        target_category = None

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                                target_category=target_category,
                                eigen_smooth=args.eigen_smooth,
                                aug_smooth=args.aug_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        
        visual_path = os.path.join(args.visual_path,args.methods)
        name = ''.join(file_name)
        # print(visual_path)
        # print(file_name)
        # print(name)

        if not(os.path.exists(visual_path)):
            os.makedirs(visual_path)

        cv2.imwrite(os.path.join(visual_path, name), cam_image)


