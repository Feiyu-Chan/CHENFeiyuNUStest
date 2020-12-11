import numpy as np
import cv2
from shapely.geometry import box
from shapely.ops import cascaded_union
from PIL import Image
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as F
import os
import json
# from utils.util import calc_iou, calc_intersection
import cv2
import random


def one_hot_encode(integer_encoding, num_classes):
    """ One hot encode.
	"""
    onehot_encoded = [0 for _ in range(num_classes)]
    onehot_encoded[integer_encoding] = 1
    return torch.tensor(onehot_encoded)


class VrdBase(Dataset):
    """VRD dataset."""
    def __init__(self, dataset_path, num_classes, type, transform=None, ifselected=False):
        # read annotations file
        self.ifselected = ifselected

        if self.ifselected:
            with open(os.path.join(dataset_path, 'vr_selected', f'vr_selected_{type}.json'), 'r') as f:
                self.annotations = json.load(f)
            self.root = os.path.join(dataset_path, 'vr_selected', f'vr_selected_{type}')
        else:
            with open(os.path.join(dataset_path, 'json_dataset', f'annotations_{type}.json'), 'r') as f:
                self.annotations = json.load(f)
            self.root = os.path.join(dataset_path, 'sg_dataset', f'sg_{type}_images')

        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class VRDDataset(VrdBase):
    """VRD dataset."""
    def __init__(self, dataset_path, type, num_class=100, transform=None, ifselected=False, ifimage=False):
        super(VRDDataset, self).__init__(dataset_path=dataset_path, num_classes=num_class, type=type, transform=transform, ifselected=ifselected)
        self.datas = []
        self.ifimage = ifimage
        for ann in self.annotations:
            for ind in range(len(self.annotations[ann])):
                self.datas.append((ann,ind))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        img_name, ind = self.datas[idx]

        img_path = os.path.join(self.root, img_name)

        img = Image.open(img_path)
        data = self.annotations[img_name][ind]
        predicate = data['predicate']
        sub = data['subject']
        obj = data['object']  # YMIN YMAX XMIN XMAX
        sub_bbox = [int(sub['bbox'][2]), int(sub['bbox'][0]), int(sub['bbox'][3]), int(sub['bbox'][1])]
        obj_bbox = [int(obj['bbox'][2]), int(obj['bbox'][0]), int(obj['bbox'][3]), int(obj['bbox'][1])]

        # takes union of sub and obj
        polygons = [box(sub_bbox[0], sub_bbox[1], sub_bbox[2], sub_bbox[3]), box(obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3])]
        unioned = cascaded_union(polygons)
        unioned = unioned.bounds
        xmin_unioned, ymin_unioned, xmax_unioned, ymax_unioned = unioned

        # crop image
        sub_img = img.crop((int(sub['bbox'][2]), int(sub['bbox'][0]), int(sub['bbox'][3]), int(sub['bbox'][1])))
        obj_img = img.crop((int(obj['bbox'][2]), int(obj['bbox'][0]), int(obj['bbox'][3]), int(obj['bbox'][1])))
        union_img = img.crop((int(xmin_unioned), int(ymin_unioned), int(xmax_unioned), int(ymax_unioned)))

        sub_img = self.transform(sub_img)
        obj_img = self.transform(obj_img)
        union_img = self.transform(union_img)
        img = self.transform(img)

        sub_target = one_hot_encode(sub['category'], 100)
        obj_target = one_hot_encode(obj['category'], 100)

        predicate_target = one_hot_encode(predicate, 46)

        if self.ifimage:
            out = (img, sub_img, sub_target, obj_img, obj_target, union_img, predicate_target)
        else:
            out = (sub_img, sub_target, obj_img, obj_target, union_img, predicate_target)

        return out


class VRDbbxDataset(VrdBase):
    """VRD dataset."""
    def __init__(self, dataset_path, type, num_class=100, transform=None, ifseleted=False):
        super(VRDbbxDataset, self).__init__(dataset_path=dataset_path, num_classes=num_class, type=type, transform=transform, ifselected=ifseleted)

        self.bbxdata = []
        for ann in self.annotations:
            bbx = []
            for data in self.annotations[ann]:
                for item in ['object', 'subject']:
                    if data[item]['bbox'] not in bbx:
                        self.bbxdata.append({'img_name': ann, 'category': data[item]['category'], 'bbox': data[item]['bbox']})
                        bbx.append(data[item]['bbox'])

    def __len__(self):
        return len(self.bbxdata)

    def __getitem__(self, idx):
        data = self.bbxdata[idx]

        img_path = os.path.join(self.root, data['img_name'])
        img = Image.open(img_path)

        bbox = data['bbox']  # YMIN,YMAX,XMIN,XMAX
        crop_img = img.crop((bbox[2], bbox[0], bbox[3], bbox[1]))

        crop_img = self.transform(crop_img)
        target = one_hot_encode(data['category'], self.num_classes)

        return (crop_img, target)



class VRDDetDataset(VrdBase):
    """VRD Detection dataset."""
    def __init__(self, dataset_path, type, num_class=100, transform=None, ifselected=False, ifimage=True):
        super(VRDDetDataset, self).__init__(dataset_path=dataset_path, num_classes=num_class, type=type, transform=transform, ifselected=ifselected)
        self.datas = []
        self.ifimage = ifimage
        for ann in self.annotations:
            self.datas.append(((ann,self.annotations[ann])))
            # for ind in range(len(self.annotations[ann])):
            #     self.datas.append((ann,ind))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        img_name, data = self.datas[idx]
        # data=self.annotations[img_name][ind]

        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path)
        img = self.transform(img)

        sub_target = []
        sub_bbox = []
        obj_target = []
        obj_bbox = []
        predicate_target = []
        for i in range(len(data)):
            sub = data[i]['subject']
            obj = data[i]['object']  # YMIN YMAX XMIN XMAX
            sub_bbox.append([int(sub['bbox'][2]), int(sub['bbox'][0]), int(sub['bbox'][3]), int(sub['bbox'][1])])
            obj_bbox.append([int(obj['bbox'][2]), int(obj['bbox'][0]), int(obj['bbox'][3]), int(obj['bbox'][1])])

            sub_target.append(one_hot_encode(sub['category'], 100))
            obj_target.append(one_hot_encode(obj['category'], 100))

            predicate = data[i]['predicate']

            predicate_target.append(one_hot_encode(predicate, 46))

        if self.ifimage:
            out = (img, sub_target, sub_bbox, obj_target, obj_bbox, predicate_target)
        else:
            out = (sub_target, sub_bbox, obj_target, obj_bbox, predicate_target)

        return out


class VRDFRDataset(VrdBase):
    def __init__(self, dataset_path, num_classes, type, cls_transform, dtc_transform, ifselected, ifdtc=False):
        super(VRDFRDataset, self).__init__(dataset_path=dataset_path, num_classes=num_classes, type=type, transform=cls_transform, ifselected=ifselected,)
        self.datas = []
        self.ifdtc=ifdtc
        self.dtc_transform = dtc_transform

        for ann in self.annotations:
            if len(self.annotations[ann])!=0:
                self.datas.append([ann, len(self.annotations[ann])])

    def __getitem__(self, idx):
        img_name, obj_num = self.datas[idx]
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path).convert('RGB')

        sub_images = []
        obj_images = []
        sub_targets = []
        obj_targets = []
        predicates = []
        union_images = []
        data = self.annotations[img_name]
        boxes = []
        labels = []
        for ind in range(obj_num):
            sub = data[ind]['subject']
            obj = data[ind]['object']
            sub_bbox = [int(sub['bbox'][2]), int(sub['bbox'][0]), int(sub['bbox'][3]), int(sub['bbox'][1])]
            obj_bbox = [int(obj['bbox'][2]), int(obj['bbox'][0]), int(obj['bbox'][3]), int(obj['bbox'][1])]

            polygons = [box(sub_bbox[0], sub_bbox[1], sub_bbox[2], sub_bbox[3]),
                        box(obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3])]
            unioned = cascaded_union(polygons)
            xmin_unioned, ymin_unioned, xmax_unioned, ymax_unioned = unioned.bounds
            sub_img = img.crop((int(sub['bbox'][2]), int(sub['bbox'][0]), int(sub['bbox'][3]), int(sub['bbox'][1])))
            obj_img = img.crop((int(obj['bbox'][2]), int(obj['bbox'][0]), int(obj['bbox'][3]), int(obj['bbox'][1])))
            union_img = img.crop((int(xmin_unioned), int(ymin_unioned), int(xmax_unioned), int(ymax_unioned)))
            sub_img = self.transform(sub_img)
            obj_img = self.transform(obj_img)
            union_img = self.transform(union_img)
            sub_images.append(sub_img)
            obj_images.append(obj_img)
            union_images.append(union_img)
            predicates.append(data[ind]['predicate'])
            sub_targets.append(sub['category'])
            obj_targets.append(obj['category'])

            for item in ['object', 'subject']:
                if data[ind][item]['bbox'] not in boxes:
                    boxes.append(data[ind][item]['bbox'])
                    labels.append(data[ind][item]['category'])

        bboxes = []
        for ind in range(len(boxes)):
            bboxes.append([int(boxes[ind][2]), int(boxes[ind][0]), int(boxes[ind][3]), int(boxes[ind][1])])
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        # print(bboxes.dtype)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        labels = torch.Tensor(labels)
        # print(labels.dtype)


        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        if self.ifdtc==True:
            target['labels']+=1
        target["image_id"] = torch.tensor([idx], dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)

        if self.dtc_transform is not None:

            img, target = self.dtc_transform(img,target)
           # print(img.size())

        #print(img.size(),len(sub_images),len(obj_images),len(union_images),len(sub_targets),len(obj_targets),len(predicates))

        return img, target, sub_images, obj_images, union_images, sub_targets, obj_targets, predicates

    def __len__(self):
        return len(self.datas)

    def collate_fn(self,batch):
        imgs=[]
        targets=[]
        for item in batch:
            imgs.append(item[0])
            targets.append(item[1])
        return [imgs,targets]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image =F.to_tensor(image)
        return image, target

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)



class VRDFR2Dataset(VrdBase):
    def __init__(self, dataset_path, num_classes, type, cls_transform, dtc_transform, ifselected, ifdtc=False):
        super(VRDFR2Dataset, self).__init__(dataset_path=dataset_path, num_classes=num_classes, type=type, transform=cls_transform, ifselected=ifselected,)
        self.datas = []
        self.ifdtc=ifdtc
        self.dtc_transform = dtc_transform

        for ann in self.annotations:
            if len(self.annotations[ann])!=0:
                self.datas.append([ann, len(self.annotations[ann])])

    def __getitem__(self, idx):
        img_name, obj_num = self.datas[idx]
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path).convert('RGB')

        sub_bboxes = []
        obj_bboxes = []
        sub_targets = []
        obj_targets = []
        predicates = []
        data = self.annotations[img_name]
        boxes = []
        labels = []
        for ind in range(obj_num):
            sub = data[ind]['subject']
            obj = data[ind]['object']
            sub_bbox = [int(sub['bbox'][2]), int(sub['bbox'][0]), int(sub['bbox'][3]), int(sub['bbox'][1])]
            obj_bbox = [int(obj['bbox'][2]), int(obj['bbox'][0]), int(obj['bbox'][3]), int(obj['bbox'][1])]
            sub_bboxes.append(sub_bbox)
            obj_bboxes.append(obj_bbox)
            obj_targets.append(obj['category'])
            sub_targets.append(sub['category'])
            predicates.append(data[ind]['[predicate'])

            for item in ['object', 'subject']:
                if data[ind][item]['bbox'] not in boxes:
                    boxes.append(data[ind][item]['bbox'])
                    labels.append(data[ind][item]['category'])

        bboxes = []
        for ind in range(len(boxes)):
            bboxes.append([int(boxes[ind][2]), int(boxes[ind][0]), int(boxes[ind][3]), int(boxes[ind][1])])

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        sub_bboxes = torch.as_tensor(sub_bboxes, dtype=torch.float32)
        obj_bboxes = torch.as_tensor(obj_bboxes, dtype=torch.float32)
        sub_targets = torch.as_tensor(sub_targets, dtype=torch.float32)
        obj_targets = torch.as_tensor(obj_targets, dtype=torch.float32)
        predicates = torch.as_tensor(predicates, dtype=torch.float32)

        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        labels = torch.Tensor(labels)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        if self.ifdtc==True:
            target['labels']+=1
        target["image_id"] = torch.tensor([idx], dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)

        if self.dtc_transform is not None:

            img, target = self.dtc_transform(img,target)
           # print(img.size())

        #print(img.size(),len(sub_images),len(obj_images),len(union_images),len(sub_targets),len(obj_targets),len(predicates))

        return img, target, sub_bboxes, obj_bboxes, sub_targets, obj_targets, predicates

    def __len__(self):
        return len(self.datas)

    def collate_fn(self,batch):
        imgs=[]
        targets=[]
        sub_bboxes=[]
        obj_bboxes = []
        obj_targets = []
        sub_targets= []
        predicates = []
        for item in batch:
            imgs.append(item[0])
            targets.append(item[1])
            sub_bboxes.append(item[2])
            obj_bboxes.append(item[3])
            sub_targets.append(item[4])
            obj_targets.append(item[5])
            predicates.append(item[6])

        return [imgs,targets,sub_bboxes, obj_bboxes, sub_targets, obj_targets, predicates]


















if __name__ == '__main__':
    img_path="./sg_dataset/sg_train_images/311670451_bb4160309c_b.jpg"
    # import cv2
    # a=cv2.imread(img_path)
    # plt.imshow(a)
    # plt.show()

    img = Image.open(img_path).convert('RGB')
    # train_trans = transforms.Compose(
    #     [transforms.Resize((224, 224)),
    #      transforms.RandomHorizontalFlip(),
    #      transforms.RandomRotation(degrees=(-30, 30)),
    #      transforms.ToTensor(),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #      ])
    # dataset = VRDFRDataset(dataset_path = './', type ='train', num_classes=100, cls_transform=train_trans, ifselected=False, dtc_transform=get_transform(train=True))
    # data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    # for i in data_loader:
    #     print(len(i),i[0].size())
    # trans = transforms.Compose(
    #     [transforms.Resize((224, 224)),
    #      transforms.ToTensor(),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #      ])
    #
    # dataset = VRDDataset("./", 'train', 100, trans)
    # loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    # print(len(dataset))
    # for i in loader:
    #     a, b, c, d, e, f = i
    #     print(type(a), type(b), type(c), type(d), type(e), type(f))
    #     print(a.size(), len(b), c.size(), len(d), e.size(), len(f))
    #     print(e[0].size())
    #     plt.imshow(e.permute(0,2,3,1)[0].numpy())
    #     plt.show()
    #     plt.imshow(a.permute(0,2,3,1)[0].numpy())
    #     plt.show()

