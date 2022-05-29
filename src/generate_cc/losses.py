import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score,f1_score,recall_score, accuracy_score, roc_auc_score

def calculate_acc(labels, preds):
    count = 0.0
    num = labels.shape[0] * labels.shape[1]
    for l, p in zip(labels, preds):
        for x, y in zip(l, p):
            if x == y:
                count += 1.0

    return count/num

class Loss_fn(torch.nn.Module):
    def __init__(self, args, mode='train'):
        super(Loss_fn, self).__init__()
        
        # self.crition = BCEWithLogitsLoss()
        self.c_crition = CrossEntropyLoss()
    

    def forward(self, model_outputs, labels):
        
        logits, c_logits = model_outputs
        labels = torch.tensor(labels).cuda()
        # loss1 = self.crition(logits, labels)
        labels_count = labels.sum(dim=1)-1
        four = torch.ones_like(labels_count)*8
        labels_count = torch.where(labels_count > 8, four, labels_count)
        # print(c_logits.shape)
        # print(labels_count.shape)
        loss2 = self.c_crition(c_logits, labels_count)

        log_p_y = F.log_softmax(logits, dim=1) # num_query x num_class
        loss1 = - labels * log_p_y
        loss1 = loss1.mean()
        loss = loss1 + 0.1 * loss2

        _, count_pred  = torch.max(c_logits, 1, keepdim=True)
        labels_count = labels_count.cpu().detach()
        count_pred = count_pred.cpu().detach()
        c_acc = accuracy_score(labels_count, count_pred)
        

        sorts, indices = torch.sort(log_p_y, descending=True)  #按行从大到小排序
        
        x = []
        for i, t in enumerate(count_pred):
            x.append(log_p_y[i][indices[i][count_pred[i][0]]])
        x = torch.tensor(x).view(log_p_y.shape[0], 1).cuda()
        one = torch.ones_like(log_p_y)
        zero = torch.zeros_like(log_p_y)
        y_pred = torch.where(log_p_y >= x, one, log_p_y)
        y_pred = torch.where(y_pred < x, zero, y_pred)

        # print(y_pred)
        # print(y_pred)
        # print(re)
        target_mode = 'macro'

        labels = labels.cpu().detach()
        y_pred = y_pred.cpu().detach()
        p = precision_score(labels, y_pred, average=target_mode)
        r = recall_score(labels, y_pred, average=target_mode)
        f = f1_score(labels, y_pred, average=target_mode)
        acc = accuracy_score(labels, y_pred)
    

        
        return loss, p, r, f, acc, y_pred
        






