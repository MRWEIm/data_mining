import torch
import utils
import argparse
import pandas as pd
import numpy as np
import pandas as pd
import random
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

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

def mutual_info(x, y):
    return mutual_info_score(x, y)


def compute_taskwise_label_distribution(y_true):
    """
    y_true: List of 4 lists, each inner list contains labels for one task
    Returns: List[Dict[str, float]] for each task
    """
    label_set = ["Yes", "No", "To some extent"]
    distributions = []

    for task_labels in y_true:
        count = Counter(task_labels)
        total = sum(count.values())
        dist = {label: count.get(label, 0) / total for label in label_set}
        distributions.append(dist)

    return distributions


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_data_path', type=str, default='./BEA/data/mrbench_v3_devset.json')
    parser.add_argument('--test_data_path', type=str, default='./BEA/data/mrbench_v3_testset.json')
    parser.add_argument('--model_path', type=str, default='./model/')
    parser.add_argument('--model_name', type=str, default='Qwen3-Embedding-4B')
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-6)

    parser.add_argument('--lstm_hid_dim', type=int, default=768)
    parser.add_argument('--mul_att_out_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=2560)
    return parser.parse_args()


class BEA_Dataset(Dataset):
    def __init__(self, conversation, response, label, args):
        self.conversation = conversation
        self.response = response
        self.label = label

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(args.model_path + args.model_name).to(device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path + args.model_name, use_fast=False)
        self.MAX_PARAGRAPHS = args.max_para
        self.HIDDEN_DIM = self.model.config.hidden_size

    def __len__(self):
        return len(self.conversation)
    
    def __getitem__(self, index):
        conversation_tensor = self.get_text_tensor(self.conversation[index], text_type='conversation')
        response_tensor = self.get_text_tensor(self.response[index], text_type='response')

        return conversation_tensor, response_tensor, self.label[index]

    def get_text_tensor(self, text, text_type):
        assert text_type in ['conversation', 'response']

        para_vectors = []
        for para in text:
            inputs = self.tokenizer(para, return_tensors="pt", truncation=True, padding=False, max_length=1024).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: [1, hidden]
            para_vectors.append(cls_embedding.squeeze(0))       # shape: [hidden]
        
        if text_type == 'conversation':
            while len(para_vectors) < self.MAX_PARAGRAPHS:
                para_vectors.append(torch.zeros(self.HIDDEN_DIM).to(self.device))
            
        para_tensor = torch.stack(para_vectors[:self.MAX_PARAGRAPHS])
        return para_tensor


def main():
    args = init_args()
    dev_set = utils.load_data(args.dev_data_path)

    task_list = ['Mistake_Identification', 'Mistake_Location', 'Providing_Guidance', 'Actionability', 'Tutor_Identity', 'First_Four', 'All']
    label_space = ['Yes', 'To some extent', 'No']
    conversation_history, tutor_responses, label = utils.data_process(dev_set, task_type=task_list[5])

    distributions = compute_taskwise_label_distribution(label)
    for i, dist in enumerate(distributions):
        print(f"Task {i}: {dist}")
    print('='*50)

    label = utils.label_convert(label)
    conversation_history = torch.load(f'./BEA/tensor/{args.model_name}_conversation_tensor.pt')
    tutor_responses = torch.load(f'./BEA/tensor/{args.model_name}_response_tensor.pt')

    dataset = TensorDataset(conversation_history, tutor_responses, label)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_rst = {
        0: {'ex_macro_f1': 0},
        1: {'ex_macro_f1': 0},
        2: {'ex_macro_f1': 0},
        3: {'ex_macro_f1': 0}
    }

    fold_avg_best_rst = {
        0: [],
        1: [],
        2: [],
        3: []
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
        loss_model = lf.AutomaticWeightedLoss(num=8)
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
            for ii in range(4):
                if best_rst[ii]['ex_macro_f1'] < eval_rst[0]['ex_macro_f1']:
                    best_rst['ex_macro_f1'] = eval_rst[0]['ex_macro_f1']
                    best_rst['eval_rst'] = eval_rst

        for ii in range(4):
            fold_avg_best_rst[ii].append(fold_best_rst[ii])

    for ii in range(4):
        avg = 0
        for jj in range(5):
            avg += fold_avg_best_rst[ii][jj]['ex_macro_f1']
        avg = avg / 5
        print(f'Task_{ii} best_avg_macro_f1: {avg:.4f}')
        print(f'Task_{ii} best_rst: {fold_avg_best_rst[ii][jj]["eval_rst"]}')

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
    print(f'avg_loss: {avg_loss:.4f}')


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