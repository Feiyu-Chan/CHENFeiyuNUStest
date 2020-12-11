import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision

import torch.optim as Optim
from torch.utils.data import Dataset, DataLoader
from VGG import VGG
import os
from dataset import VRDFRDataset,get_transform
import visdom
import json
from loss import FocalLoss
from torch.optim import lr_scheduler
import detection.utils
from detection.engine4relaiton import train_one_epoch, evaluate
from collections import OrderedDict

GPU = [0]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in GPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
else:
    device = torch.device("cpu")

backbone = torchvision.models.vgg16(pretrained=False)
vgg_16 = VGG(num_class=46, pretrained=False, ifbatch=True).to(device)
print("Loading pretrained weights from %s" % ('./checkpoints/task2_2/vgg4clsrlt.pth'))
vgg_16.load_state_dict(torch.load('./checkpoints/task2_2/vgg4clsrlt.pth'))
print("Loading pretrained weights from %s" % ('./checkpoints/vgg16_caffe.pth'))
state_dict = torch.load('./checkpoints/vgg16_caffe.pth')
backbone.load_state_dict({k: v for k, v in state_dict.items() if k in backbone.state_dict()})

backbone = backbone.features
backbone.out_channels = 512
detection_model = torchvision.models.detection.FasterRCNN(backbone, num_classes = 100).to(device)
params = [p for p in detection_model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001,)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=20,
                                               gamma=0.1)
train_trans = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(degrees=(-30, 30)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
test_trans = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])

dataset = VRDFRDataset(dataset_path = './', type ='train', num_classes=100, cls_transform=train_trans, ifselected=False, dtc_transform=get_transform(train=True))
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

test_dataset = VRDFRDataset(dataset_path
                            = './', type ='test', num_classes=100, cls_transform=test_trans, ifselected=True, dtc_transform=get_transform(train=False))
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)






for epoch in range(20):
    # train for one epoch, printing every 10 iterations
    _, detection_model = train_one_epoch(detection_model,
                                         rlt_model=vgg_16, optimizer=optimizer, data_loader=data_loader,
                                         device=device, epoch=epoch, print_freq=10)
    torch.save(detection_model.state_dict(), './checkpoints/task2_1/' + 'Faster-rcnn' + '_%d.pth' % (epoch))
    torch.save(detection_model.state_dict(), './checkpoints/task2_1/' + 'Faster-rcnn' + '_latest.pth')

    # # update the learning rate
    ##evaluate(detection_model, test_data_loader, device=device)

    print("That's it!")

