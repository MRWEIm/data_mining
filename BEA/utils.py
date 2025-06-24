import json
import re
import torch
import nltk
from tqdm import tqdm

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file) 
    return data


def data_process(data, task_type='Mistake_Identification'):
    assert task_type in ['Mistake_Identification', 'Mistake_Location', 'Providing_Guidance', 'Actionability', 'Tutor_Identity', 'First_Four', 'All']
    conversation_history, tutor_responses = [], []
    label = {'Mistake_Identification': [], 'Mistake_Location': [], 'Providing_Guidance': [], 'Actionability': [], 'Tutor_Identity': []}
    
    for d in data:
        conversation_history.extend([d['conversation_history']] * len(d['tutor_responses']))

        for key, value in d['tutor_responses'].items():
            tutor_responses.append(value['response'])
            label['Mistake_Identification'].append(value['annotation']['Mistake_Identification'])
            label['Mistake_Location'].append(value['annotation']['Mistake_Location'])
            label['Providing_Guidance'].append(value['annotation']['Providing_Guidance'])
            label['Actionability'].append(value['annotation']['Actionability'])
            label['Tutor_Identity'].append(key)
    if task_type == 'All':
        return conversation_history, tutor_responses, label
    elif task_type == 'First_Four':
        return conversation_history, tutor_responses, [label['Mistake_Identification'], label['Mistake_Location'], label['Providing_Guidance'], label['Actionability']]
    return conversation_history, tutor_responses, label[task_type]


def label_convert(label):
    label_tensor = []
    for ii in range(len(label)):
        label_tensor.append(torch.tensor([0 if l == 'No' else (1 if l == 'To some extent' else 2) for l in label[ii]]))
    return torch.stack(label_tensor, dim=1)

def split_into_sentence(content):
    return nltk.sent_tokenize(content)


def text_process(text, text_type):
    text_paras = []
    max_para = 0
    for t in tqdm(text):
        if text_type == 'conversation':
            pattern = r'(Tutor|Student):([\s\S]*?)(?=(Tutor|Student):|$)'
            matches = re.findall(pattern, t)
            paras = [f"{speaker}:{content.strip()}" for speaker, content, _ in matches]
        elif text_type == 'response':
            paras = [t]
        max_para = max(max_para, len(paras))
        text_paras.append(paras)

    return text_paras, max_para


