from sklearn.metrics import confusion_matrix
import numpy as np
# y_true = [[1, 0, 1, 0, 0, 1,],[0,1,0,1,1,0]]
# y_pred = [[1, 1, 1, 1, 0, 1,],[1,0,0,0,1,1]]
#
#
#
# pre = 0.0
# recall6 =0.0
#
# for i in range(2):
#     cm = confusion_matrix(y_true[i], y_pred[i])
#
#     recall = np.diag(cm) / np.sum(cm, axis=1)
#     precision = np.diag(cm) / np.sum(cm, axis=0)
#     recall6 += np.mean(recall)
#     print(recall)
#     # print(precision)
#     pre += np.mean(precision)
# print(recall6 / 2, pre / 2)
import torch
# from einops import rearrange, repeat
# x = rearrange(torch.randn(1,2048,16,16), 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4)
# print(x.size())
import  dgl
import networkx as nx


# full_g = dgl.from_networkx(nx.complete_graph(3))
# full_g.ndata['feat'] = torch.randn(3,10)
# full_g.edata['feat'] = torch.randn(6,3)

# g = dgl.DGLGraph()
# g.add_nodes(3)
# g.ndata['feat'] = torch.randn(3,10)
#
#
# g.add_edges([0,1,2],[1,2,3])
# g.edata['feat'] = torch.randn(3,3)


# import torch.nn as nn
# emb = nn.Embedding(3,6)
# y = emb(torch.tensor([[0,1,2],[0,1,2]]))
# print(y)

# src = torch.randn(2,256,16,16)
# bs, c, h, w = src.shape #bs,256,16,16
# src = src.flatten(2).permute(2, 0, 1)#256,1,1

# pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #256,1,256

# query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) #19,bs,256
# print(torch.__version__)
# a = 4
# print('4444'+a)
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
def glove_vec():
    # glove_input_file = 'glove.840B.300d.txt'
    # word2vec_output_file = 'glove.840B.300d.word2vec.txt'
    # glove2word2vec(glove_input_file, word2vec_output_file)
    path = 'GoogleNews-vectors-negative300.bin'

    glove_model = KeyedVectors.load_word2vec_format(path, binary=True)

    cat_vec = glove_model['Cheek']
    print(np.array(cat_vec))

    print(glove_model.most_similar('Cheek'))

import math
import pickle
def word_vec():
    path = '/data/huiminwang/dataset/GoogleNews-vectors-negative300.bin'
    emb = []

    model = KeyedVectors.load_word2vec_format(path, binary=True)
    word2vec = []
    with open('Celeba_train.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.split()
            vec = None
            for word in words:
                if vec is not None:
                    vec += model[word].copy()
                else:
                    vec = model[word].copy()
            vec = vec/len(words)*1.0
            word2vec.append(vec)
        word2vec = np.array(word2vec)

        sum_fz = 0
        distance_cos = 0
        sum_vec1 = 0
        sum_vec2 = 0
        sum_fm = 0
        for i in range(len(vec)):
            sum_fz += word2vec[8][i] * word2vec[9][i]
            sum_fm += math.sqrt(math.pow(word2vec[8][i], 2)) + math.sqrt(math.pow(word2vec[9][i], 2))

        distance_cos = sum_fz / sum_fm
        print(distance_cos)  # 余弦值
        print(math.acos(distance_cos))  # 角度值

    # print(word2vec)
    # with open('word2vec.pkl', 'wb+') as f:
    #     pickle.dump(word2vec, f)




    # embeds = nn.Embedding(40, 96)
    # word2vec = torch.tensor([])
    # for i in range(40):
    #     lookup_tensor = torch.tensor([word_to_ix[line[i]]], dtype=torch.long)
    #     # embed = embeds(lookup_tensor)
    #     word2vec = torch.cat((word2vec, embed), 0)

    # word2vec = word2vec.detach().numpy()

    # with open('dataset/emb.pkl', 'wb+') as f:
    #     pickle.dump(word2vec, f)

if __name__ == '__main__':
    # glove_vec()
    word_vec()