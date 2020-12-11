import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def compute_iou(rec1, rec2):
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0

def Calculate_Iou(pred_bbox, target_bbox):
    target_bbox_np = [target_bbox[0][0].numpy(), target_bbox[1][0].numpy(),
                      target_bbox[2][0].numpy(), target_bbox[3][0].numpy()]
    judge = False
    iou = compute_iou(pred_bbox, target_bbox_np)
    # print("...........")
    # print(pred_bbox)
    # print(target_bbox_np)
    # print(iou)
    # print("...........")
    if float(iou) >= 0.5:
        judge = True
    return judge


# Without calculating bbox iou.
def Eval_Recall(eval_thr, sub_input_ori, relation_input_ori, obj_input_ori, sub_target,
                relation_target, obj_target, index_target, judge_pred):
    """
    :param eval_thr: value = 50 when evaluation in R@50 ; value = 100 when evaluation in R@100
    """
    batchsize = len(sub_input_ori)

    sub_max = []
    obj_max = []
    rel_max = []

    sub_input = []
    relation_input = []
    obj_input = []

    for i in range(batchsize):
        sub_input.append(F.softmax(sub_input_ori[i], dim=1))
        relation_input.append(F.softmax(relation_input_ori[i], dim=1))
        obj_input.append(F.softmax(obj_input_ori[i], dim=1))

        _, sub_max_input = torch.sort(sub_input[i], dim=1, descending=True)
        sub_max.append(sub_max_input)
        _, obj_max_input = torch.sort(obj_input[i], dim=1, descending=True)
        obj_max.append(obj_max_input)
        _, rel_max_input = torch.sort(relation_input[i], dim=1, descending=True)
        rel_max.append(rel_max_input)


    recall_count = 0

    tag_target = torch.zeros(1000)
    for b in range(batchsize):
        rank_score = torch.zeros(1000)
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    rank_score[i*100+j*10+k] += sub_input[b][0][sub_max[b][0][i]] *\
                                                relation_input[b][0][rel_max[b][0][j]] *\
                                                obj_input[b][0][obj_max[b][0][k]]

        rank_max = torch.argsort(rank_score, dim=0, descending=True)

        _, sub_target_max = torch.max(sub_target[b], 1)
        _, rel_target_max = torch.max(relation_target[b], 1)
        _, obj_target_max = torch.max(obj_target[b], 1)

        sub_index = (sub_max[b][0].cuda()==sub_target_max.cuda()).nonzero()
        rel_index = (rel_max[b][0].cuda()==rel_target_max.cuda()).nonzero()
        obj_index = (obj_max[b][0].cuda()==obj_target_max.cuda()).nonzero()

        this_score = (rank_max.cuda() == (sub_index[0]*100 + rel_index[0]*10 + obj_index[0])).nonzero()
        if this_score.size() != 0 and this_score.size()[0] != 0:
            if (this_score[0] < eval_thr) and tag_target[index_target[b]] == 0:
                recall_count += 1
                tag_target[index_target[b]] = 1
                if eval_thr == 100:
                    judge_pred[b] = 1



    return recall_count

