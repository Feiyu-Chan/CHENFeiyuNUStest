### for focal loss test
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torch.optim as Optim
from torch.utils.data import Dataset, DataLoader
from VGG import VGG
import os
from dataset import VRDDataset
import visdom
import json
from collections import OrderedDict
from loss import FocalLoss
from torch.optim import lr_scheduler




class task1_clsrlt ():
    def __init__(self, save_path='./checkpoints/task1_clsrlt/', device=None):
        self.save_path = save_path
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.batchsize = 32
        self.iterations = 20000
        self.test_acc = {'test_acc': []}
        self.model_name = 'vgg4clsrlt'
        self.log_name = 'test_log'
        self.num_class = 46
        self.ifbatch = True

    def train_clsrlt(self,ifmypretianed = False):

        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)

        if ifmypretianed:
            vgg_16 = VGG(num_class=self.num_class, pretrained=False, ifbatch=self.ifbatch).to(self.device)
            vgg_16.load_state_dict(torch.load(self.save_path + self.model_name + '_%d.pth'%(25)))
        else:
            vgg_16 = VGG(num_class=self.num_class, pretrained=True, ifbatch=self.ifbatch).to(self.device)

        # optimizer = Optim.Adam(vgg_16.parameters(), lr=0.0001, betas=(0.5, 0.9))
        optimizer = Optim.SGD(vgg_16.parameters(), lr = 1e-3)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        trans = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(degrees=(-30, 30)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])

        dataset = VRDDataset("./", 'train', self.num_class, trans, ifselected=True)
        loader = DataLoader(dataset, batch_size=self.batchsize, shuffle=True, num_workers=0)
        print("start trainning...")
        # criterion = nn.CrossEntropyLoss().to(device)
        criterion = FocalLoss(class_num=self.num_class).to(self.device)
        train_iter = 0
        epoch = 0
        while train_iter <= self.iterations:
            running_loss = 0.0
            running_acc = 0.0
            vgg_16.train()
            for _, _, _, _, union_image, predicate_target in loader:

                union_image = union_image.to(self.device)
                predicate_target = predicate_target.to(self.device)
                output = vgg_16(union_image)

                loss = criterion(output, torch.max(predicate_target, 1)[1])
                running_loss += loss.item()*predicate_target.size(0)

                _, pred = torch.max(output, 1)  # 预测最大值所在的位置标签
                correct = (pred == torch.max(predicate_target, 1)[1]).float().mean()
                num_correct = (pred == torch.max(predicate_target, 1)[1]).float().sum()
                running_acc += num_correct.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_iter += 1
                print('Finish {} iter, Loss: {:.6f}, Acc: {:.6f}'.
                      format(train_iter, loss, correct))
            epoch += 1
            print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.
                          format(epoch, running_loss / len(dataset), running_acc / len(dataset)))
            torch.save(vgg_16.state_dict(), self.save_path + self.model_name+'_%d.pth'%(epoch))
            torch.save(vgg_16.state_dict(), self.save_path + self.model_name+'_latest.pth')
            self.test_clsrlt(ifstore = True)
            scheduler.step()


    def test_clsrlt(self,ifstore = True, load_name = None):
        test_vgg_16 = VGG(num_class=self.num_class, pretrained=False, ifbatch=self.ifbatch).to(self.device)
        if load_name is None:
            self.load_name = self.save_path+ self.model_name + '_latest.pth'
        else:
            self.load_name = load_name
        test_vgg_16.load_state_dict(torch.load(self.load_name))
        # model_dict = torch.load(self.load_name)
        # new_state_dict = OrderedDict()
        # for k, v in model_dict.items():
        #     name = k[7:]
        #     new_state_dict[name] = v
        # test_vgg_16.load_state_dict(new_state_dict)


        trans = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])
        test_dataset = VRDDataset("./", 'test', self.num_class, trans, ifselected=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
        acc_list = []
        print('start testing')
        test_vgg_16.eval()
        for _, _, _, _, union_image, predicate_target in test_loader:

            union_iamge = union_image.to(self.device)
            predicate_target = predicate_target.to(self.device)
            output = test_vgg_16(union_iamge)
            #iter += 1
            _, pred = torch.max(output, 1)  # 预测最大值所在的位置标签
            acc = (pred == torch.max(predicate_target, 1)[1]).float()
            #print('Testing Image {}, Acc: {:.6f}'.format(iter, acc.item()))
            acc_list.append(acc.item())
        aver_acc = torch.mean(torch.Tensor(acc_list))
        print('Totaol tested image: {}, average_acc: {:.6f}'.format(iter, aver_acc.item()))
        if ifstore == True:
            self.test_acc['test_acc'].append(aver_acc.item())
            tl = open(self.save_path + self.log_name, 'w')
            json.dump(self.test_acc, fp=tl)



if __name__ == '__main__':
    GPU = [2]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        device = torch.device("cpu")


    experiment = task1_clsrlt(save_path='./checkpoints/task1_clsrlt/', device=device)
    experiment.train_clsrlt(ifmypretianed=False)