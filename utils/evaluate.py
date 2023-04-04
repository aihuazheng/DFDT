# --------------------------------------------------------------------
# This file is part of
# Weakly-supervised Pedestrian Attribute Localization Network.
#
# Weakly-supervised Pedestrian Attribute Localization Network
# is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Weakly-supervised Pedestrian Attribute Localization Network
# is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Weakly-supervised Pedestrian Attribute Localization Network.
# If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------
#import sklearn
import numpy as np
from easydict import EasyDict

#对每个属性进行评估
def mA_LB(attr, gt):
    num = attr.__len__()
    num_attr = attr[0].__len__() # 26
    # challenging = []
    # acc_collect = []

    attr = np.array(attr).astype(float)
    gt = np.array(gt).astype(float)

    mA = []
    for i in range(num_attr):
        acc = (sum([attr[j][i] * gt[j][i] for j in range(num)]) / (sum([gt[j][i] for j in range(num)])) + sum(
            [(1 - attr[j][i]) * (1 - gt[j][i]) for j in range(num)]) / (
                   sum([(1 - gt[j][i]) for j in range(num)]))) / 2
        mA.append('%.4f' % acc)

    return mA


def mA(attr, gt):
    num = attr.__len__() 
    print("$$$$$$$$num######="+str(num)) #-->batch的数量
    num_attr = attr[0].__len__()
    print("########num_attr######"+str(num_attr)) #--》26
    challenging = []
    acc_collect = []

    attr = np.array(attr).astype(float)
    gt = np.array(gt).astype(float)

    mA = 0
    # for i in xrange(num_attr):
    #     print '--------------------------------------------'
    #     print i
    #     print sum([attr[j][i] for j in xrange(num)]), \
    #         ':', sum([attr[j][i] * gt[j][i] for j in xrange(num)]), \
    #         ':', sum([gt[j][i] for j in xrange(num)])
    #     print sum([attr[j][i] * gt[j][i] for j in xrange(num)]) / sum([gt[j][i] for j in xrange(num)])
    #     print sum([(1 - attr[j][i]) for j in xrange(num)]), \
    #         ':', sum([(1 - attr[j][i]) * (1 - gt[j][i]) for j in xrange(num)]), \
    #         ':', sum([(1 - gt[j][i]) for j in xrange(num)])
    #     print sum([(1 - attr[j][i]) * (1 - gt[j][i]) for j in xrange(num)]) / sum([(1 - gt[j][i]) for j in xrange(num)])
    #
    #     acc = (sum([attr[j][i] * gt[j][i] for j in xrange(num)]) / (sum([gt[j][i] for j in xrange(num)]) + 1e-10) + sum(
    #         [(1 - attr[j][i]) * (1 - gt[j][i]) for j in xrange(num)]) / (sum([(1 - gt[j][i]) for j in xrange(num)]) + 1e-10)) / 2
    #     mA += acc
    # #     acc_collect.append(acc)
    # #     print acc
    # #     if acc < 0.75:
    # #         challenging.append(i)
    # mA = mA / num_attr
    mA = (sum([(
            sum([attr[j][i] * gt[j][i] for j in range(num)])
            / (sum([gt[j][i] for j in range(num)]) + 1e-10)
            + sum([(1 - attr[j][i]) * (1 - gt[j][i]) for j in range(num)])
            / (sum([(1 - gt[j][i]) for j in range(num)]) + 1e-10)
    ) for i in range(num_attr)])) / (2 * num_attr)
    return mA
def mA1(pt_result,gt_result):
    eps = 1e-20
    mA = 0
    gt_pos = np.sum((gt_result == 1).astype(float), axis=0)
    gt_neg = np.sum((gt_result == 0).astype(float), axis=0)
    pt_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=0)
    pt_neg = np.sum((gt_result == 0).astype(float) * (pt_result == 0).astype(float), axis=0)
    label_pos_acc = 1.0 * pt_pos / (gt_pos+eps)
    label_neg_acc = 1.0 * pt_neg / (gt_neg+eps)
    label_acc = (label_pos_acc + label_neg_acc) / 2
    mA = np.mean(label_acc)
    return mA

def example_based(attr, gt):
    num = attr.__len__()
    num_attr = attr[0].__len__()
    # print num, num_attr

    acc = 0
    prec = 0
    rec = 0
    f1 = 0

    sk_acc = 0
    sk_prec = 0
    sk_recall = 0
    sk_f1 = 0

    attr = np.array(attr).astype(bool)
    gt = np.array(gt).astype(bool)

    for i in range(num):
        intersect = sum((attr[i] & gt[i]).astype(float))
        union = sum((attr[i] | gt[i]).astype(float))
        attr_sum = sum((attr[i]).astype(float)) # 
        gt_sum = sum((gt[i]).astype(float)) # gt_pos

        acc += intersect / (union + 1e-10)
        prec += intersect / (attr_sum + 1e-10)
        rec += intersect / (gt_sum + 1e-10)

        # sk_acc += sklearn.metrics.jaccard_similarity_score(gt[i], attr[i])
        # sk_prec += sklearn.metrics.precision_score(gt[i], attr[i])
        # sk_recall += sklearn.metrics.recall_score(gt[i], attr[i])
        # sk_f1 += sklearn.metrics.f1_score(gt[i], attr[i])

    acc /= num
    prec /= num
    rec /= num
    f1 = 2 * prec * rec / (prec + rec)

    # sk_acc /= num
    # sk_prec /= num
    # sk_recall /= num
    # sk_f1 /= num
    return acc, prec, rec, f1

def example_based1(pt_result,gt_result):
    result = {}
    eps = 1e-20
    ###############################
    # label metrics
    # TP + FN
    # gt_pos = np.sum((gt_result == 1), axis=0).astype(float)
    # # TN + FP
    # gt_neg = np.sum((gt_result == 0), axis=0).astype(float)
    # # TP
    # true_pos = np.sum((gt_result == 1) * (pt_result == 1), axis=0).astype(float)
    # # TN
    # true_neg = np.sum((gt_result == 0) * (pt_result == 0), axis=0).astype(float)
    # # FP
    # false_pos = np.sum(((gt_result == 0) * (pt_result == 1)), axis=0).astype(float)
    # # FN
    # false_neg = np.sum(((gt_result == 1) * (pt_result == 0)), axis=0).astype(float)

    # label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    # label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # # mean accuracy
    # label_ma = (label_pos_recall + label_neg_recall) / 2

    gt_pos = np.sum((gt_result == 1).astype(float), axis=0)
    gt_neg = np.sum((gt_result == 0).astype(float), axis=0)
    pt_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=0)
    pt_neg = np.sum((gt_result == 0).astype(float) * (pt_result == 0).astype(float), axis=0)
    label_pos_acc = 1.0 * pt_pos / gt_pos
    label_neg_acc = 1.0 * pt_neg / gt_neg
    label_acc = (label_pos_acc + label_neg_acc) / 2
    result['label_pos_acc'] = label_pos_acc
    result['label_neg_acc'] = label_neg_acc
    result['label_ma'] = np.mean(label_acc)

    gt_pos=np.sum((gt_result==1).astype(float),axis=1)
    pt_pos=np.sum((pt_result==1).astype(float),axis=1)
    floatersect_pos=np.sum((gt_result==1).astype(float)*(pt_result==1).astype(float),axis=1)
    union_pos=np.sum(((gt_result==1)+(pt_result==1)).astype(float),axis=1)
    cnt_eff=float(gt_result.shape[0])
    for iter,key in enumerate(gt_pos):
        if key==0:
            union_pos[iter]=1
            pt_pos[iter]=1
            gt_pos[iter]=1
            cnt_eff=cnt_eff-1
            continue
        if pt_pos[iter]==0:
            pt_pos[iter]=1
    instance_acc=np.sum(floatersect_pos/union_pos)/cnt_eff
    instance_precision=np.sum(floatersect_pos/pt_pos)/cnt_eff
    instance_recall=np.sum(floatersect_pos/gt_pos)/cnt_eff
    floatance_F1=2*instance_precision*instance_recall/(instance_precision+instance_recall)
    result['instance_acc']=instance_acc
    result['instance_precision']=instance_precision
    result['instance_recall']=instance_recall
    result['instance_F1']=floatance_F1
    return result



def get_pedestrian_metrics(preds_probs,gt_label, threshold=0.5):
    # pred_label = preds_probs > threshold

    eps = 1e-20
    result = EasyDict()

    ###############################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (preds_probs == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (preds_probs == 0), axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (preds_probs == 1)), axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (preds_probs == 0)), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((preds_probs == 1), axis=1).astype(float)
    # true positive
    intersect_pos = np.sum((gt_label == 1) * (preds_probs == 1), axis=1).astype(float)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (preds_probs == 1)), axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps)
    instance_prec = intersect_pos / (true_pos + eps)
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    instance_f1 = np.mean(instance_f1)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    return result

if __name__ == '__main__':
    x = np.array([[0,1,0],[1,0,1],[1,0,1]])
    y = np.array([[0,1,1], [1,1,1],[1,0,1]]) #gt
    # x = np.array([[0,1,0]])
    # y = np.array([[0,1,1]])
    out =(x==y)
    print(out)
    out = out.sum(0)
    print(out/3)
    out = example_based(x,y)
    out =mA(x,y)
    print(out)