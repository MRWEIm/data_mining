import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight


def task_sim_loss(y_true, y_pred):
    return 

def total_loss(y_true, y_pred, crf=None, weight_task=None):
    """
    y_true: Tensor, shape [batch_size, 4], int labels (0,1,2)
    y_pred: Tensor, shape [batch_size, 4, 3], raw logits
    """
    if weight_task is None:
        weight_task = torch.tensor([2.0, 4.0, 0.3], dtype=torch.float32).to(y_pred.device)

    total_loss = []
    for task_id in range(4):
        pred_task = y_pred[:, task_id, :]   # [B, 3]
        label_task = y_true[:, task_id]     # [B]

        loss_fn = nn.CrossEntropyLoss(weight=weight_task)
        task_loss = loss_fn(pred_task, label_task)

        alpha = 1.0
        total_loss.append(alpha * task_loss)

    # for task_id in range(4):
    #     pred_task = crf[:, task_id, :]   # [B, 3]
    #     label_task = y_true[:, task_id]     # [B]

    #     loss_fn = nn.CrossEntropyLoss(weight=weight_task)
    #     task_loss = loss_fn(pred_task, label_task)

    #     alpha = 1.0
    #     total_loss.append(alpha * task_loss)

    total_loss.append(crf.mean())
    return total_loss
    return 0.3 * crf.mean() + total_loss / 4.0
    return total_loss / 4.0

def per_class_accuracy(y_true, y_pred, num_classes=3):
    acc_dict = {}

    for cls in range(num_classes):
        # 找到属于该类的索引
        cls_indices = np.where(y_true == cls)[0]
        if len(cls_indices) == 0:
            acc_dict[cls] = None  # 没有该类样本，无法计算准确率
            continue
        # 计算该类的准确率
        cls_acc = accuracy_score(y_true[cls_indices], y_pred[cls_indices])
        acc_dict[cls] = round(float(cls_acc), 4)

    return acc_dict


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        # 创建一个长度为 num 的张量，元素初始化为 1，且支持梯度更新
        params = torch.ones(num, requires_grad=True)  
        # 将张量封装为 PyTorch 的 Parameter，使其成为模型可学习参数
        self.params = torch.nn.Parameter(params)

    def forward(self, x):
        loss_sum = 0  
        # 遍历输入的损失（x 是包含各任务损失的可变参数）
        for i, loss in enumerate(x):  
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)  
        return loss_sum  

def evaluate(references, predictions):
    if len(predictions.shape) > 2:
        predictions = np.argmax(predictions, axis=-1) 

    eval_rst = []
    for ii in range(predictions.shape[1]):
        acc_dict = per_class_accuracy(references[:, ii], predictions[:, ii])
        # print(f'task_{ii+1}: {acc_dict}')

        ex_acc = accuracy_score(references[:, ii], predictions[:, ii])
        ex_f1 = f1_score(references[:, ii], predictions[:, ii], average='macro')

        y_true_len = np.where(references[:, ii] == 0, 0, 1)
        y_pred_len = np.where(predictions[:, ii] == 0, 0, 1)
        len_acc = accuracy_score(y_true_len, y_pred_len)
        len_f1 = f1_score(y_true_len, y_pred_len, average='macro')

        eval_rst.append({
            'ex_macro_f1': round(float(ex_f1), 4),
            'ex_acc': round(float(ex_acc), 4),
            'len_macro_f1': round(float(len_f1), 4),
            'len_acc': round(float(len_acc), 4)
        })

    return eval_rst