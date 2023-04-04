import os
project_index = os.getcwd().find('fine-grained2019AAAI')
root = os.getcwd()[0:project_index] + 'fine-grained2019AAAI'
import sys
sys.path.append(root)
import torch
from torch import nn
import numpy as np
import random
from scipy.stats import bernoulli as bn

############    可选择的    #####################
class SelectDropMAX(nn.Module):

    def __init__(self, pk=0.5, supression= "join",mask_height=7,mask_width=2):
        """
        实现论文中的diversificationblock, 接受一个三维的feature map，返回一个numpy的列表，作为遮罩
        :param pk: pk是bc'中随机遮罩的概率
        :param r: bc''中行分成几块
        :param c: bc''中列分成几块
        """
        super(SelectDropMAX, self).__init__()
        self.supression = supression
        self.pk = pk
        self.mask_width = mask_width
        self.mask_height = mask_height
    def helperb1(self,feature_map):
        ##############对每个通道上的HW做mask,置信度最高的位置，mask置为1############################
        row, col = torch.where(feature_map == torch.max(feature_map)) #标记出每个hw特征图最大置信度的位置
        b1 = torch.zeros_like(feature_map).cuda()
        # b1 = np.zeros(feature_map.size(0),feature_map.size(1))
        for i in range(len(row)):
            r, c = int(row[i]), int(col[i])
            b1[r, c] = 1
            # b1 = torch.from_numpy(b1).cuda()
            # b1 = b1.data.cpu()
        return b1  ##tensor
    def create_mask(self,feature_map, mask_height,mask_width,x=None, y=None):
        # 黑色是0  白色缺失部分是255   实际使用时，需要将255变为1。 因为需要和原图做加减运算
        height, width = feature_map.size()
        mask = torch.zeros_like((feature_map)).cuda() # 生成一个覆盖全部大小的全黑mask
        mask_x = x if x is not None else random.randint(0, width - mask_width) # 缺失部分左下角的x坐标
        mask_y = y if y is not None else random.randint(0, height - mask_height)  #缺失部分左上角的y坐标
        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1  # 将中间缺失白色部分标为1
        # mask = torch.from_numpy(mask).cuda() #numpy-->tensor
        # print("######mask#####"+str(mask))
        return mask   #tensor

    # def from_num_to_block(self,mat, r, c, num):
    #     #####  resnet mat[h,w],,,,r,c指的是每行分几个块，每列分为几个块
    #     assert len(mat.shape) == 2, ValueError("Feature map shape is wrong!")
    #     mat = mat.data.cpu()
    #     res = np.zeros_like(mat)
            
    #     row, col = mat.shape
    #     block_r, block_c = int(row / r), int(col / c) #patch size
    #     index = np.arange(r * c) + 1 #1,2,3,4,......,12
    #     index = index.reshape(r, c)#将一行的索引reshape成3行4列
    #     index_r, index_c = np.argwhere(index == num)[0] #索引为num的下标
    #     # print(index_r, index_c) #0,2
    #     if index_c + 1 == c:
    #             end_c = c + 1
    #     else:
    #         end_c = (index_c + 1) * block_c #3
    #     if index_r + 1 == r:
    #         end_r = r + 1
    #     else:
    #         end_r = (index_r + 1) * block_r #1
    #     # print(end_c,end_r)
    #     res[index_r * block_r: end_r, index_c * block_c:end_c] = 1
    #     return res

    def forward(self, x):
        """
        x---(bs,c,h,w) ([16, 26, 7, 7])
        """
        batch_supression = []
        sample_maps_list = torch.split(x, 1)
        ### 按照batch循环
        for sample_map in sample_maps_list:
            sample_map = sample_map.squeeze(0)
            
            ###########   对每个样本进行处理 ###########
            if len(sample_map.shape) == 3:
                resb1 = []
                resb2 = []
                feature_maps_list = torch.split(sample_map, 1)
                ###   按照通道循环
                for feature_map in feature_maps_list:
                    # feature_map = feature_map.squeeze()
                    # tmp = self.select_drop(feature_map) #peak mask
                    feature_map = feature_map.squeeze(0)
                    if self.supression == "Peak":
                        tmp = self.helperb1(feature_map) #peak mask
                        resb1.append(tmp) #[tensor,tensor..]
                    if self.supression == "Random":
                        tmp1 = self.create_mask(feature_map,self.mask_height,self.mask_width) #patch mask
                        resb2.append(tmp1)
                    if self.supression == "Join":
                        tmp = self.helperb1(feature_map) #peak mask
                        resb1.append(tmp)
                        tmp1 = self.create_mask(feature_map,self.mask_height,self.mask_width) #patch mask
                        resb2.append(tmp1)
                        
                # resb1=torch.stack(resb1,0)
                resb2=torch.stack(resb2,0)

            elif len(sample_map.shape) == 2:
                if self.supression == "Peak":
                    tmp = self.helperb1(sample_map) #peak mask
                    resb1.append(tmp)
        
                if self.supression == "Random":
                    tmp1 = self.create_mask(sample_map,self.mask_width,self.mask_height) #patch mask
                    resb2.append(tmp1)
                    
                if self.supression == "Join":
                    tmp = self.helperb1(sample_map) #peak mask
                    resb1.append(tmp)
                  
                    tmp1 = self.create_mask(sample_map,self.mask_width,self.mask_height) #patch mask
                    resb2.append(tmp1)

                # 需要手动修改：Peak使用resb1,Random使用resb2,Join两个都使用
                resb1=torch.stack(resb1,0)
                resb2=torch.stack(resb2,0) 
                   
            else:
                raise ValueError
            res_features = []
            if len(sample_map.shape) == 3:
                for x in range(len(resb1)):
                    if self.supression == "Peak":
                        index_block = torch.clamp(resb1[x],0,1)  #每个通道的
                        res_feature = sample_map[x] - 0.9* torch.mul(sample_map[x],index_block.cuda())
                        res_features.append(res_feature)
                    if self.supression == "Random":
                        index_block = torch.clamp(resb2[x],0,1)  #每个通道的
                        res_feature = sample_map[x] - 0.9* torch.mul(sample_map[x],index_block.cuda())
                        # res_features.append(res_feature.data.cpu().numpy())
                        res_features.append(res_feature)
                    if self.supression == "Join":
                        index_block = torch.clamp(resb1[x]+resb2[x],0,1)  #每个通道的
                        res_feature = sample_map[x] - 0.9* torch.mul(sample_map[x],index_block.cuda())
                        res_features.append(res_feature)
                res_features = torch.stack(res_features,0)
                    
               
            elif len(sample_map.shape) == 2:
                if self.supression == "Peak":
                    index_block = torch.clamp(resb1[x],0,1)  #每个通道的
                    res_features = sample_map[x] - 0.9* torch.mul(sample_map[x],index_block.cuda())
                if self.supression == "Random":
                    index_block = torch.clamp(resb2[x],0,1)  #每个通道的
                    res_features = sample_map[x] - 0.9* torch.mul(sample_map[x],index_block.cuda())
                if self.supression == "Join":
                    index_block = torch.clamp(resb1[x]+resb2[x],0,1)  #每个通道的
                    res_features = sample_map[x] - 0.9* torch.mul(sample_map[x],index_block.cuda())

                    # index_block = torch.clamp(resb1[x],0,1)  #每个通道的
                    # res_feature = sample_map[x] - 0.9* torch.mul(sample_map[x],index_block.cuda())
            batch_supression.append(res_features)
        batch_supression = torch.stack(batch_supression,0)
        return batch_supression


if __name__ == '__main__':
    feature_maps = torch.rand([16,3,3,4])
    print("feature maps is: ", feature_maps)
    db = SelectDropMAX()
    db.cuda()
    res = db(feature_maps.cuda())
    print("################")
    print(feature_maps)
    print(res, len(res))





