import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision

import torch.optim as Optim
from torch.utils.data import Dataset, DataLoader
from VGG import VGG
import os
from dataset import VRDDataset, VRDDetDataset
import visdom
import json
from loss import FocalLoss
from torch.optim import lr_scheduler
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltimage
from eval_recall import Eval_Recall, Calculate_Iou
from shapely.geometry import box
from shapely.ops import cascaded_union
import cv2 as cv

import torch.nn.functional as F

GPU = [2]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in GPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
else:
    device = torch.device("cpu")

trans = transforms.Compose(
    [# transforms.Resize([256, 256]),
     transforms.ToTensor(),
     #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])

trans_vgg = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
detection_model.to(device)
detection_model.eval()

############ load two classifier models ###################
test_vgg_obj = VGG(num_class=100, pretrained=False, ifbatch=True).to(device)
test_vgg_rlt = VGG(num_class=46, pretrained=False, ifbatch=True).to(device)
test_vgg_obj.load_state_dict(torch.load('./checkpoints/task1_comb/vgg4clsobj.pth'))
test_vgg_rlt.load_state_dict(torch.load('./checkpoints/task1_comb/vgg4clsrlt.pth'))
test_vgg_obj.eval()
test_vgg_rlt.eval()


detdataset = VRDDetDataset("./", 'test', 100, transform=trans, ifselected=True, ifimage=True)
detloader = DataLoader(detdataset, batch_size=1, shuffle=False, num_workers=0)

Recall_50_count = 0
Recall_100_count = 0
det_union_num = 0
all_relationship_count = 0


for image, sub_target, sub_bbox, obj_target, obj_bbox, predicate_target, img_name in detloader:
    # if img_name[0] != '3665708190_b99175077d_o.jpg' and img_name[0] != '4418514401_cd86bc8e53_b.jpg':
    #      continue
    faster_rcnn_ratio = 2
    image_input = []
    image_r = nn.functional.interpolate(image, size=[int((image.size()[2])/faster_rcnn_ratio),
                                                     int((image.size()[3])/faster_rcnn_ratio)],
                                      mode="bilinear")

    image_r = image_r.reshape([image_r.size()[1],image_r.size()[2],image_r.size()[3]])
    image_r = image_r.to(device)
    image_input.append(image_r)
    with torch.no_grad():
        out = detection_model(image_input)
    boxes = out[0]['boxes']
    image = image.reshape([image.size()[1], image.size()[2], image.size()[3]])

    pred_bbox_num = boxes.cpu().shape[0]
    score_threshold = 0.01

    pred_union_sub_bbox = []
    pred_union_obj_bbox = []
    gt_union_sub_category = []
    gt_union_obj_category = []
    gt_union_relationship = []
    gt_union_index = []


    for i in range(pred_bbox_num):
        if out[0]['scores'][i] > score_threshold:
            xmin = round(boxes[i][0].item())
            ymin = round(boxes[i][1].item())
            xmax = round(boxes[i][2].item())
            ymax = round(boxes[i][3].item())

            pred_bbox_sub = [xmin*faster_rcnn_ratio, ymin*faster_rcnn_ratio,
                             xmax*faster_rcnn_ratio, ymax*faster_rcnn_ratio]

            for j in range(len(sub_target)):
                if Calculate_Iou(pred_bbox_sub, sub_bbox[j]):
                    for k in range(pred_bbox_num):
                        if out[0]['scores'][k] > score_threshold:
                            xmin = round(boxes[k][0].item())
                            ymin = round(boxes[k][1].item())
                            xmax = round(boxes[k][2].item())
                            ymax = round(boxes[k][3].item())

                            # pred_bbox_obj = [xmin, xmax, ymin, ymax]
                            pred_bbox_obj = [xmin*faster_rcnn_ratio, ymin*faster_rcnn_ratio,
                                             xmax*faster_rcnn_ratio, ymax*faster_rcnn_ratio]

                            if Calculate_Iou(pred_bbox_obj, obj_bbox[j]):
                                pred_union_sub_bbox.append(pred_bbox_sub)
                                pred_union_obj_bbox.append(pred_bbox_obj)

                                gt_union_sub_category.append(sub_target[j])
                                gt_union_obj_category.append(obj_target[j])
                                gt_union_relationship.append(predicate_target[j])
                                gt_union_index.append(j)

    pred_union_sub_category = []
    pred_union_obj_category = []
    pred_union_relationship = []

    det_object_num = len(pred_union_sub_bbox)

    for i in range(det_object_num):
        #############transform#########
        pred_bbox_sub = pred_union_sub_bbox[i]
        pred_bbox_obj = pred_union_obj_bbox[i]

        polygons = [box(pred_bbox_sub[0], pred_bbox_sub[1],
                        pred_bbox_sub[2], pred_bbox_sub[3]),
                    box(pred_bbox_obj[0], pred_bbox_obj[1],
                        pred_bbox_obj[2], pred_bbox_obj[3])]
        unioned = cascaded_union(polygons)
        unioned = unioned.bounds
        xmin_unioned, ymin_unioned, xmax_unioned, ymax_unioned = unioned

        # crop image
        union_img = image[:, int(ymin_unioned):int(ymax_unioned),
                    int(xmin_unioned):int(xmax_unioned)]

        union_img = torch.unsqueeze(union_img, dim=0)
        union_img_resize = nn.functional.interpolate(union_img, size=[224, 224], mode="bilinear")

        union_img_resize = union_img_resize.to(device)
        with torch.no_grad():
            pred_relationship = test_vgg_rlt(union_img_resize)
        # save of probalities vector of the relationship
        pred_union_relationship.append(pred_relationship)

        torch.cuda.empty_cache()
        ##############add transform###############

    for i in range(det_object_num):
        crop_img_sub = image[:, pred_union_sub_bbox[i][1]:pred_union_sub_bbox[i][3],
                   pred_union_sub_bbox[i][0]:pred_union_sub_bbox[i][2]]
        crop_img_sub = trans_vgg(crop_img_sub)
        crop_img_sub = torch.unsqueeze(crop_img_sub, dim=0)
        # save of probabilities vector of the object
        crop_img_sub_resize = nn.functional.interpolate(crop_img_sub, size=[224, 224], mode="bilinear")
        crop_img_sub_resize = crop_img_sub_resize.to(device)
        with torch.no_grad():
            pred_object_sub = test_vgg_obj(crop_img_sub_resize)
        pred_union_sub_category.append(pred_object_sub)

        torch.cuda.empty_cache()

    for i in range(det_object_num):

        crop_img_obj = image[:, pred_union_obj_bbox[i][1]:pred_union_obj_bbox[i][3],
                   pred_union_obj_bbox[i][0]:pred_union_obj_bbox[i][2]]
        crop_img_obj = trans_vgg(crop_img_obj)
        crop_img_obj = torch.unsqueeze(crop_img_obj, dim=0)

        crop_img_obj_resize = nn.functional.interpolate(crop_img_obj, size=[224, 224], mode="bilinear")
        crop_img_obj_resize = crop_img_obj_resize.to(device)
        with torch.no_grad():
            pred_object_obj = test_vgg_obj(crop_img_obj_resize)
        pred_union_obj_category.append(pred_object_obj)

        torch.cuda.empty_cache()

    this_R50_count = 0
    this_R100_count = 0

    judge_R100 = torch.zeros(len(pred_union_sub_category))

    if len(pred_union_sub_category) != 0:
        this_R50_count = Eval_Recall(50, pred_union_sub_category, pred_union_relationship, pred_union_obj_category,
                    gt_union_sub_category, gt_union_relationship, gt_union_obj_category, gt_union_index, judge_R100)
        Recall_50_count += this_R50_count
        this_R100_count = Eval_Recall(100, pred_union_sub_category, pred_union_relationship, pred_union_obj_category,
                    gt_union_sub_category, gt_union_relationship, gt_union_obj_category, gt_union_index, judge_R100)
        Recall_100_count += this_R100_count

    # image_show = pltimage.imread('./vr_selected/vr_selected_test/' + img_name[0])
    # object_name = json.load(open('./json_dataset/objects.json'))
    # category_name = json.load(open('./json_dataset/vr_selected_predicates.json'))
    # plt.imshow(image_show)
    #
    # show_time = 0
    # for i in range(len(pred_union_sub_category)):
    #     if judge_R100[i] == 1:
    #         show_time += 1
    #         if show_time == 1:
    #             plt.gca().add_patch(plt.Rectangle(xy=(pred_union_sub_bbox[i][0], pred_union_sub_bbox[i][1]),
    #                                           width=pred_union_sub_bbox[i][2] - pred_union_sub_bbox[i][0],
    #                                           height=pred_union_sub_bbox[i][3] - pred_union_sub_bbox[i][1],
    #                                           edgecolor=[1, 0, 1],
    #                                           fill=False, linewidth=1))
    #             plt.gca().add_patch(plt.Rectangle(xy=(pred_union_sub_bbox[i][0], pred_union_sub_bbox[i][1]),
    #                                           width=pred_union_sub_bbox[i][2] - pred_union_sub_bbox[i][0],
    #                                           height=pred_union_sub_bbox[i][3] - pred_union_sub_bbox[i][1],
    #                                           edgecolor=[1, 0, 1],
    #                                           fill=False, linewidth=1))
    #             _, pred = torch.max(gt_union_sub_category[i], 1)  # 预测最大值所在的位置标签
    #             pred_union_sub_score = F.softmax(pred_union_sub_category[i], dim=1)
    #             plt.text(pred_union_sub_bbox[i][0], pred_union_sub_bbox[i][1],
    #                      " "+object_name[int(pred.cpu())]+" "+str(pred_union_sub_score[0][pred][0].cpu().numpy()),
    #                      color='purple', verticalalignment='bottom')
    #
    #             plt.gca().add_patch(plt.Rectangle(xy=(pred_union_obj_bbox[i][0]-3, pred_union_obj_bbox[i][1]-3),
    #                                           width=pred_union_obj_bbox[i][2] - pred_union_obj_bbox[i][0],
    #                                           height=pred_union_obj_bbox[i][3] - pred_union_obj_bbox[i][1],
    #                                           edgecolor=[1, 0, 0],
    #                                           fill=False, linewidth=1))
    #             _, pred = torch.max(gt_union_obj_category[i], 1)  # 预测最大值所在的位置标签
    #             pred_union_obj_score = F.softmax(pred_union_obj_category[i], dim=1)
    #             plt.text(pred_union_obj_bbox[i][0], pred_union_obj_bbox[i][1],
    #                      " "+object_name[int(pred.cpu())]+" "+str(pred_union_obj_score[0][pred][0].cpu().numpy()),
    #                      color='red', verticalalignment='top')
    #
    #             _, pred = torch.max(gt_union_relationship[i], 1)  # 预测最大值所在的位置标签
    #             pred_union_rel_score = F.softmax(pred_union_relationship[i], dim=1)
    #             plt.text(0, 0, " "+category_name[int(pred.cpu())]+" "+str(pred_union_rel_score[0][pred][0].cpu().numpy()),
    #                      color='black', verticalalignment='top')
    #
    #
    # plt.show()


    all_relationship_count += len(predicate_target)

    det_union_num += det_object_num

    print("_____________________________")
    print("Detected objects nums = ", det_object_num)
    print("This R@50 count= ", this_R50_count)
    print("This R@100 count= ", this_R100_count)
    print("This GT count= ", len(predicate_target))


print("########################################################")
print("Test done.")
print("Detected union objects nums =", det_union_num)
print("Recall @50 count =", Recall_50_count)
print("Recall @100 count =", Recall_100_count)
print("GT relationship count =", all_relationship_count)
print("Recall @50 =", Recall_50_count / all_relationship_count)
print("Recall @100 =", Recall_100_count / all_relationship_count)

