import torch
import utils
import argparse
import pandas as pd
import numpy as np
import pandas as pd
import json
import loss_fn as lf
from itertools import product
from collections import Counter
from tqdm import tqdm
from BEA import BEA
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sklearn.model_selection import KFold
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_data_path', type=str, default='./BEA/data/mrbench_v3_devset.json')
    parser.add_argument('--test_data_path', type=str, default='./BEA/data/mrbench_v3_testset.json')
    parser.add_argument('--model_path', type=str, default='./model/')
    parser.add_argument('--model_name', type=str, default='Qwen3-Embedding-4B')
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-5)

    parser.add_argument('--lstm_hid_dim', type=int, default=768)
    parser.add_argument('--mul_att_out_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=2560)
    return parser.parse_args()


def main():
    args = init_args()
    dev_set = utils.load_data(args.dev_data_path)

    task_list = ['Mistake_Identification', 'Mistake_Location', 'Providing_Guidance', 'Actionability', 'Tutor_Identity', 'First_Four', 'All']
    label_space = ['Yes', 'To some extent', 'No']
    conversation_history, tutor_responses, label = utils.data_process(dev_set, task_type=task_list[5])

    label = utils.label_convert(label)
    conversation_history = torch.load(f'./BEA/tensor/{args.model_name}_conversation_tensor.pt')
    tutor_responses = torch.load(f'./BEA/tensor/{args.model_name}_response_tensor.pt')

    dataset = TensorDataset(conversation_history, tutor_responses, label)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    learning_rate = [5e-5, 1e-5, 5e-6]
    batch_size = [32, 64, 128]

    for lr, bs in list(product(learning_rate, batch_size)):
        args.learning_rate = lr
        args.batch_size = bs

        print(f'lr: {lr}, bs: {bs}')
        print('='*50)

        five_fold_best_avg = {
            0: [],
            1: [],
            2: [],
            3: []
        }

        best_rst = {
            0: {'ex_macro_f1': 0},
            1: {'ex_macro_f1': 0},
            2: {'ex_macro_f1': 0},
            3: {'ex_macro_f1': 0}
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f'Fold_{fold} starting...')
            print('='*50)
            train_subset = Subset(dataset, train_idx)
            test_subset = Subset(dataset, val_idx)
            train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_subset, batch_size=args.batch_size)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = BEA(args=args, output_dim=4).to(device)
            loss_model = lf.AutomaticWeightedLoss(num=5)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

            fold_best_rst = {
                0: {'ex_macro_f1': 0},
                1: {'ex_macro_f1': 0},
                2: {'ex_macro_f1': 0},
                3: {'ex_macro_f1': 0}
            }

            for _ in range(args.epochs):
                train(model, optimizer, train_dataloader, device, loss_model=loss_model)
                eval_rst = eval(model, test_dataloader, device)
                for ii in range(4):
                    if eval_rst[ii]['ex_macro_f1'] > fold_best_rst[ii]['ex_macro_f1']:
                        fold_best_rst[ii]['ex_macro_f1'] = eval_rst[ii]['ex_macro_f1']
                        fold_best_rst[ii]['eval_rst'] = eval_rst
                    if eval_rst[ii]['ex_macro_f1'] > best_rst[ii]['ex_macro_f1']:
                        best_rst[ii]['ex_macro_f1'] = eval_rst[ii]['ex_macro_f1']
                        best_rst[ii]['eval_rst'] = eval_rst
                    
            for ii in range(4):
                five_fold_best_avg[ii].append(fold_best_rst[ii])

        with open(f'./BEA/result/lr_{lr}_bs_{bs}_results.json', 'w') as f:
            json.dump([five_fold_best_avg, best_rst], f, indent=4)


    # for ii in range(4):
    #     total = 0
    #     for jj in range(5):
    #         total += five_fold_best_avg[ii][jj]['ex_macro_f1']
    #     avg = total / 5
    #     print(f'Task_{ii} best_macro_f1: {avg:.4f}')

    # for ii in range(4):
    #     print(f'Task_{ii} best_rst: {best_rst[ii]}')

    return 


def train(model, optimizer, dataloader, device, loss_model, weights=None):
    model.train()

    total_loss = 0
    batch_num = 0
    for batch in dataloader:
        conversation, response, label = map(lambda x: x.to(device), batch)
        pred, crf = model(conversation, response, label)
        loss = lf.total_loss(label, pred, crf)
        loss = loss_model(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batch_num += 1
    avg_loss = total_loss / batch_num
    # print(f'avg_loss: {avg_loss:.4f}')


@torch.no_grad()
def eval(model, dataloader, device):
    model.eval()
    predictions = []
    labels = []
    for batch in dataloader:
        conversation, response, label = map(lambda x: x.to(device), batch)
        pred, crf = model(conversation, response)
        predictions.append(crf.cpu().numpy())
        labels.append(label.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    eval_rst = lf.evaluate(labels, predictions)
    # print(eval_rst)
    # print('=' * 50)
    return eval_rst



if __name__ == '__main__':
    main()