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
from loss import RankLoss, FocalLoss
from torch.optim import lr_scheduler



class task1_comb ():
    def __init__(self, save_path='./checkpoints/task1_comb/', device=None):
        self.save_path = save_path
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.batchsize = 16
        self.iterations = 20000
        self.test_acc = {'test_subject_acc': [], 'test_object_acc': [], 'test_relation_acc': []}
        self.model_name4obj = 'vgg4obj'
        self.model_name4rlt = 'vgg4rlt'
        self.log_name = 'test_log'
        self.obj_num_class = 100
        self.rlt_num_class = 46
        self.ifbatch = True
        # self.train_acc = {'subject_acc':0, 'relation_acc':0, 'object_acc':0}

    def train_comb(self):

        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)

        vgg_obj = VGG(num_class=self.obj_num_class, pretrained=False, ifbatch=self.ifbatch).to(self.device)
        vgg_rlt = VGG(num_class=self.rlt_num_class, pretrained=False, ifbatch=self.ifbatch).to(self.device)

        vgg_obj.load_state_dict(torch.load('./checkpoints/task1_comb/vgg4clsobj.pth'))
        vgg_rlt.load_state_dict(torch.load('./checkpoints/task1_comb/vgg4clsrlt.pth'))

        # obj_optimizer = Optim.Adam(vgg_obj.parameters(), lr=0.0001, betas=(0.5, 0.9))
        # rlt_optimizer = Optim.Adam(vgg_rlt.parameters(), lr=0.0001, betas=(0.5, 0.9))
        obj_optimizer = Optim.SGD(vgg_obj.parameters(), lr = 1e-3)
        rlt_optimizer = Optim.SGD(vgg_rlt.parameters(), lr = 1e-3)
        obj_scheduler = lr_scheduler.ExponentialLR(obj_optimizer, gamma=0.95)
        rlt_scheduler = lr_scheduler.ExponentialLR(rlt_optimizer, gamma=0.95)

        #optimizer = Optim.SGD(vgg_16.parameters(), lr = 1e-3)
        trans = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])

        dataset = VRDDataset("./", 'train', 100, trans, ifselected=True)
        loader = DataLoader(dataset, batch_size=self.batchsize, shuffle=True, num_workers=0)
        print("start trainning...")
        criterion_CLSo = FocalLoss(class_num=100).to(device)
        criterion_CLSr = FocalLoss(class_num=46).to(device)

        criterion_RL = RankLoss().to(self.device)
        train_iter = 0
        epoch = 0
        while train_iter <= self.iterations:
            running_loss = 0.0
            running_acc = {'subject_acc':0.0, 'relation_acc':0.0, 'object_acc':0.0}

            vgg_obj.train()
            vgg_rlt.train()
            for sub_image, sub_target, obj_image, obj_target, union_image, predicate_target in loader:
                sub_image = sub_image.to(self.device)
                sub_target = sub_target.to(self.device)
                obj_image = obj_image.to(self.device)
                obj_target = obj_target.to(self.device)
                union_image = union_image.to(self.device)
                predicate_target = predicate_target.to(self.device)

                sub_out = vgg_obj(sub_image)
                obj_out = vgg_obj(obj_image)
                rlt_out = vgg_rlt(union_image)

                _, pred = torch.max(rlt_out, 1)  # 预测最大值所在的位置标签



                loss_RL = criterion_RL(sub_out, rlt_out, obj_out,
                                 torch.max(sub_target, 1)[1],
                                 torch.max(predicate_target, 1)[1],
                                 torch.max(obj_target, 1)[1])

                loss_sub = criterion_CLSo(sub_out, torch.max(sub_target, 1)[1])
                loss_obj = criterion_CLSo(obj_out, torch.max(obj_target, 1)[1])
                loss_rlt = criterion_CLSr(rlt_out, torch.max(predicate_target, 1)[1])
                # print(loss_RL)
                # print(loss_rlt)
                # print(loss_obj)
                loss = loss_RL + loss_sub + loss_obj +loss_rlt

                running_loss += loss.item()*self.batchsize

                rlt_optimizer.zero_grad()
                obj_optimizer.zero_grad()
                loss.backward()
                rlt_optimizer.step()
                obj_optimizer.step()


                sub_acc, sub_acc_sum = self.pred_acc(sub_out, sub_target)
                rlt_acc, rlt_acc_sum = self.pred_acc(rlt_out, predicate_target)
                obj_acc, obj_acc_sum = self.pred_acc(obj_out, obj_target)

                running_acc['subject_acc'] += sub_acc_sum
                running_acc['relation_acc'] += rlt_acc_sum
                running_acc['object_acc'] += obj_acc_sum

                train_iter += 1
                print('Finish {} iter, Loss: {:.6f}, Subject_Acc: {:.6f}, Object_Acc: {:.6f}, Relation_Acc: {:.6f}'.
                      format(train_iter, loss, sub_acc, obj_acc, rlt_acc))

            epoch += 1
            print('Finish {} epoch, Loss: {:.6f}, Subject_Acc: {:.6f}, Object_Acc: {:.6f}, Relation_Acc: {:.6f}'.
                          format(epoch, running_loss/len(dataset), running_acc['subject_acc']/len(dataset),
                                 running_acc['object_acc']/len(dataset), running_acc['relation_acc']/len(dataset)))
            rlt_scheduler.step()
            obj_scheduler.step()

            torch.save(vgg_obj.state_dict(), self.save_path + self.model_name4obj + '_%d.pth'%(epoch))
            torch.save(vgg_obj.state_dict(), self.save_path + self.model_name4obj + '_latest.pth')
            torch.save(vgg_rlt.state_dict(), self.save_path + self.model_name4rlt + '_%d.pth'%(epoch))
            torch.save(vgg_rlt.state_dict(), self.save_path + self.model_name4rlt + '_latest.pth')
            self.test_comb()

    def pred_acc(self, output, target):
        _, pred = torch.max(output, 1)  # 预测最大值所在的位置标签
        correct = (pred == torch.max(target, 1)[1]).float().mean()

        num_correct = (pred == torch.max(target, 1)[1]).float().sum()
        return correct, num_correct


    def test_comb(self,ifstore = True):
        test_vgg_obj = VGG(num_class=self.obj_num_class, pretrained=False, ifbatch=self.ifbatch).to(self.device)
        test_vgg_rlt = VGG(num_class=self.rlt_num_class, pretrained=False, ifbatch=self.ifbatch).to(self.device)

        self.load_name4rlt = self.save_path + self.model_name4rlt + '_latest.pth'
        self.load_name4obj = self.save_path + self.model_name4obj + '_latest.pth'
        test_vgg_obj.load_state_dict(torch.load(self.load_name4obj))
        test_vgg_rlt.load_state_dict(torch.load(self.load_name4rlt))

        trans = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])
        test_dataset = VRDDataset("./", 'test',100, trans, ifselected=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

        sum_sub_acc = 0.0
        sum_rlt_acc = 0.0
        sum_obj_acc = 0.0

        print('start testing')
        test_vgg_obj.eval()
        test_vgg_rlt.eval()
        for sub_image, sub_target, obj_image, obj_target, union_image, predicate_target in test_loader:

            sub_image = sub_image.to(self.device)
            sub_target = sub_target.to(self.device)
            obj_image = obj_image.to(self.device)
            obj_target = obj_target.to(self.device)
            union_image = union_image.to(self.device)
            predicate_target = predicate_target.to(self.device)

            sub_out = test_vgg_obj(sub_image)
            obj_out = test_vgg_obj(obj_image)
            rlt_out = test_vgg_rlt(union_image)

            sub_acc, _ = self.pred_acc(sub_out, sub_target)
            rlt_acc, _ = self.pred_acc(rlt_out, predicate_target)
            obj_acc, _ = self.pred_acc(obj_out, obj_target)

            sum_sub_acc += sub_acc
            sum_rlt_acc += rlt_acc
            sum_obj_acc += obj_acc

        print('Total tested image, Subject_Acc: {:.6f}, Object_Acc: {:.6f}, Relation_Acc: {:.6f}'.
                  format(sum_sub_acc/len(test_dataset), sum_obj_acc/len(test_dataset), sum_rlt_acc/len(test_dataset)))

        if ifstore == True:
            self.test_acc['test_subject_acc'].append(sum_sub_acc.item() / len(test_dataset))
            self.test_acc['test_object_acc'].append(sum_obj_acc.item() / len(test_dataset))
            self.test_acc['test_relation_acc'].append(sum_rlt_acc.item() / len(test_dataset))
            tl = open(self.save_path + self.log_name, 'w')
            json.dump(self.test_acc, fp=tl)

if __name__ == '__main__':
    GPU = [3]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        device = torch.device("cpu")


    experiment = task1_comb(save_path='./checkpoints/task1_comb/', device=device)
    experiment.train_comb()