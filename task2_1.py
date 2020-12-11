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
from detection.engine import train_one_epoch, evaluate
from collections import OrderedDict
from torchvision.models.detection.rpn import AnchorGenerator

GPU = [3]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in GPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
else:
    device = torch.device("cpu")

backbone = torchvision.models.vgg16(pretrained=False)

print("Loading pretrained weights from %s" % ('./checkpoints/vgg16_caffe.pth'))
state_dict = torch.load('./checkpoints/vgg16_caffe.pth')
backbone.load_state_dict({k: v for k, v in state_dict.items() if k in backbone.state_dict()})

backbone = backbone.features
backbone.out_channels = 512

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

num_class = 100+1 # BECAUSE the background should be inclueded
detection_model = torchvision.models.detection.FasterRCNN(backbone, num_classes = num_class, rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler).to(device)
mypretrained = True
if mypretrained:
    detection_model.load_state_dict(torch.load('./checkpoints/task2_1/Faster-rcnn_9.pth'))



params = [p for p in detection_model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=10,
                                               gamma=0.5)
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

dataset = VRDFRDataset(dataset_path = './', type ='train', num_classes=100, cls_transform=train_trans, ifselected=False, ifdtc=True, dtc_transform=get_transform(train=True))
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8, collate_fn=dataset.collate_fn)

test_dataset = VRDFRDataset(dataset_path
                            = './', type ='test', num_classes=100, cls_transform=test_trans, ifselected=True, ifdtc=True, dtc_transform=get_transform(train=False))
test_data_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=8, collate_fn=test_dataset.collate_fn)

if os.path.exists('./checkpoints/task2_1/') is False:
    os.mkdir('./checkpoints/task2_1/')

for epoch in range(50):
    # train for one epoch, printing every 10 iterations
    _, detection_model = train_one_epoch(detection_model, optimizer, data_loader, device, epoch, print_freq=10)
    torch.save(detection_model.state_dict(), './checkpoints/task2_1/' + 'Faster-rcnn' + '_%d.pth' % (epoch))
    torch.save(detection_model.state_dict(), './checkpoints/task2_1/' + 'Faster-rcnn' + '_latest.pth')

    # # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(detection_model, test_data_loader, device=device)

    print("That's it!")




