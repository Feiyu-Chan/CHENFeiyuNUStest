import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)

        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss



class RankLoss(nn.Module):
    def __init__(self):
        super(RankLoss,self).__init__()

    def __call__(self, sub_input, relation_input, obj_input, sub_target, relation_target, obj_target):
        batchsize = sub_input.size(0)
        sub_input = F.softmax(sub_input, dim=1)
        relation_input = F.softmax(relation_input, dim=1)
        obj_input = F.softmax(obj_input, dim=1)

        gt=torch.zeros(batchsize).cuda()
        for b in range(batchsize):
            gt[b]+=sub_input[b,sub_target[b]]*relation_input[b,relation_target[b]]*obj_input[b,obj_target[b]]

        _,sub_max=torch.sort(sub_input,dim=1,descending=True)
        _,obj_max=torch.sort(obj_input,dim=1,descending=True)
        _,rel_max=torch.sort(relation_input,dim=1,descending=True)

        pre=torch.zeros(batchsize).cuda()
        for b in range(batchsize):
            if sub_max[b][0] == sub_target[b] and obj_max[b][0] == obj_target[b] and rel_max[b][0] == relation_target[b]:
                tmp=torch.zeros(8).cuda()
                cnt=torch.tensor(0)
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            tmp[cnt]+=sub_input[b,sub_max[b][i]]*relation_input[b,rel_max[b][j]]*obj_input[b,obj_max[b][k]]
                            cnt += 1
                _,tmp_max=torch.sort(tmp)
                pre[b]+=tmp[tmp_max[1]]
            else:
                pre[b] += sub_input[b, sub_max[b][0]] * relation_input[b, rel_max[b][0]] * obj_input[b, obj_max[b][0]]
        out=torch.tensor(1.0).cuda()-gt+pre
        margin=torch.zeros(out.size()).cuda()
        return torch.mean(torch.where(out>torch.tensor(0.0).cuda(),out,margin))

















if __name__ == '__main__':
    # crition = FocalLoss(10)
    # inputs = torch.randn(8,10)
    # target = torch.randint(10,[8])
    # print(target)
    # loss = crition(inputs, target)

    inputs = torch.randn(8, 10)
    print(inputs.size(0))
    max_value, max_idx = torch.max(inputs, dim=1)
    min_value, min_idx = torch.min(inputs, dim=1)
    print(max_idx)
    print(inputs)
    print(inputs[:, max_idx])
    # inputs[:, max_idx.item()] = min_value




