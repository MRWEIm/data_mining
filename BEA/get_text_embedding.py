import argparse
import torch
import utils
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_data_path', type=str, default='./BEA/data/mrbench_v3_devset.json')
    parser.add_argument('--test_data_path', type=str, default='./BEA/data/mrbench_v3_testset.json')
    parser.add_argument('--model_path', type=str, default='./model/')
    parser.add_argument('--model_name', type=str, default='Qwen3-Embedding-4B')
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lstm_hid_dim', type=int, default=768)
    parser.add_argument('--mul_att_out_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=1024)
    return parser.parse_args()

def get_text_tensor(text, text_type, tokenizer, model, device):
    assert text_type in ['conversation', 'response']

    para_vectors = []
    for para in text:
        inputs = tokenizer(para, return_tensors="pt", truncation=True, padding=False, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: [1, hidden]
        para_vectors.append(cls_embedding.squeeze(0))       # shape: [hidden]
    
    if text_type == 'conversation':
        while len(para_vectors) < 20:
            para_vectors.append(torch.zeros(model.config.hidden_size).to(device))
        
    para_tensor = torch.stack(para_vectors[:20])
    return para_tensor

def main():
    args = init_args()
    dev_set = utils.load_data(args.dev_data_path)

    task_list = ['Mistake_Identification', 'Mistake_Location', 'Providing_Guidance', 'Actionability', 'Tutor_Identity', 'First_Four', 'All']
    label_space = ['Yes', 'To some extent', 'No']
    conversation_history, tutor_responses, label = utils.data_process(dev_set, task_type=task_list[5])

    label = utils.label_convert(label)
    conversation_history, max_para = utils.text_process(conversation_history, text_type='conversation')
    tutor_responses, _ = utils.text_process(tutor_responses, text_type='response')
    args.max_para = max_para

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModel.from_pretrained(args.model_path + args.model_name).to(device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path + args.model_name, use_fast=False)
    print(f'embed_dim: {model.config.hidden_size}')

    c_tensor = []
    for c in tqdm(conversation_history):
        tensor = get_text_tensor(c, 'conversation', tokenizer, model, device)
        c_tensor.append(tensor)
    c_tensor = torch.stack(c_tensor, dim=0)
    torch.save(c_tensor, f'./BEA/tensor/{args.model_name}_conversation_tensor.pt')

    t_tensor = []
    for t in tqdm(tutor_responses):
        tensor = get_text_tensor(t, 'response', tokenizer, model, device)
        t_tensor.append(tensor)
    t_tensor = torch.stack(t_tensor, dim=0)
    torch.save(t_tensor, f'./BEA/tensor/{args.model_name}_response_tensor.pt')


if __name__ == '__main__':
    main()