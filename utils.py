import math
import json
import torch
import random
import datetime

import argparse
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.nn.functional as F

# class Attacker_Dataset(Dataset):
#     def __init__(self, mode, pos_data_dir, neg_data_dir, tokenizer):
#         with open(pos_data_dir, 'r') as f:
#             pos_data = json.load(f)
#         with open(neg_data_dir, 'r') as f:
#             neg_data = json.load(f)
        
#         if mode == 'train':
#             self.pos_data = pos_data[:int(0.9*len(pos_data))]
#             self.neg_data = neg_data[:int(0.9*len(neg_data))]
#         else:
#             self.pos_data = pos_data[int(0.9*len(pos_data)):]
#             self.neg_data = neg_data[int(0.9*len(neg_data)):]

#         self.label = [1] * len(self.pos_data) + [0] * len(self.neg_data)
#         self.data = self.pos_data + self.neg_data
#         # shuffle
#         random.seed(42)
#         random.shuffle(self.data)
#         random.seed(42)
#         random.shuffle(self.label)

#         self.pos_length = len(self.pos_data)
#         self.neg_length = len(self.neg_data)
#         self.total_length = len(self.data)

#         print('data length, pos length, neg length: ', self.total_length, self.pos_length, self.neg_length)
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return self.total_length
    
#     def __getitem__(self, index):
#         record = self.data[index]
#         label = self.label[index]
        
#         # 不用instruction
#         source_text = record['input']
#         # source_text = record['instruction'] + record['input']
#         target_text = record['output']
#         input_ids = self.tokenizer.encode(
#                 source_text, padding=True, truncation=True, max_length=512)
        
#         target_ids = self.tokenizer.encode(
#                 target_text, padding=True, truncation=True, max_length=512)
#         out_dict = {}
#         out_dict['input_ids'] = torch.LongTensor(input_ids)
#         out_dict['input_length'] = len(input_ids)
       
#         out_dict['target_ids'] = torch.LongTensor(target_ids)
#         out_dict['target_length'] = len(target_ids)

#         out_dict['label'] = label

#         return out_dict
    
#     def collate_fn(self, batch):
#         batch_entry = {}
#         B = len(batch)

#         S_W_L = max(entry['input_length'] for entry in batch)
#         T_W_L = max(entry['target_length'] for entry in batch)

#         input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
#         target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

#         labels = torch.ones(B, dtype=torch.float)

#         for i, entry in enumerate(batch):
#             input_ids[i, :entry['input_length']] = entry['input_ids']
#             target_ids[i, :entry['target_length']] = entry['target_ids']
#             labels[i] = entry['label']

#         word_mask = target_ids != self.tokenizer.pad_token_id
#         target_ids[~word_mask] = -100

#         batch_entry['input_ids'] = input_ids
#         batch_entry['target_ids'] = target_ids

#         batch_entry['labels'] = labels

#         return batch_entry

# def get_attacker_loader(mode, pos_path, neg_path, tokenizer, batch_size):
#     dataset = Attacker_Dataset(mode, pos_path, neg_path, tokenizer)
#     if mode == 'train':
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=True,
#             num_workers=4, collate_fn=dataset.collate_fn)
#     else:
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=False,
#             num_workers=4, collate_fn=dataset.collate_fn, drop_last=False)
#     return loader

class Random_Dataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        with open(data_dir, 'r') as f:
            self.data = json.load(f)

        self.total_length = len(self.data)
        print('data length: ', self.total_length)
        self.tokenizer = tokenizer

    def __len__(self):
        return self.total_length
    
    def __getitem__(self, index):
        record = self.data[index]
        
        # 不用instruction
        source_text = record['input']
        # source_text = record['instruction'] + record['input']
        if 'Yes' in record['output']:
            target_text = 'No.'
        elif 'No' in record['output']:
            target_text = 'Yes.'

        input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=512)
        
        target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=512)
        out_dict = {}
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
       
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        return out_dict
    
    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100

        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids

        return batch_entry


def get_random_loader(mode, path, tokenizer, batch_size):

    dataset = Random_Dataset(path, tokenizer)
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, collate_fn=dataset.collate_fn, drop_last=False)
    return loader

class My_Dataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        with open(data_dir, 'r') as f:
            self.data = json.load(f)

        self.total_length = len(self.data)
        print('data length: ', self.total_length)
        self.tokenizer = tokenizer

    def __len__(self):
        return self.total_length
    
    def __getitem__(self, index):
        record = self.data[index]
        
        # 不用instruction
        source_text = record['input']
        # source_text = record['instruction'] + record['input']
        target_text = record['output']
        input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=512)
        
        target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=512)
        out_dict = {}
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
       
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        return out_dict
    
    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100

        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids

        return batch_entry


def get_loader(mode, path, tokenizer, batch_size):

    dataset = My_Dataset(path, tokenizer)
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, collate_fn=dataset.collate_fn, drop_last=False)
    return loader


def compute_kl(pretrained_model, current_model, batch):
    normal_outputs = current_model(
        batch["input_ids"],
        labels=batch["target_ids"],
    ).logits

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"],
            labels=batch["target_ids"],
        ).logits
        pretrained_outputs = pretrained_outputs.detach()

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs, -1)

    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    return loss

def compute_forced_kl(pretrained_model, forget_model, current_model, batch, alpha):
    normal_outputs = current_model(
        batch["input_ids"],
        labels=batch["target_ids"],
    ).logits

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"],
            labels=batch["target_ids"],
        ).logits
        pretrained_outputs = pretrained_outputs.detach()
        forget_outputs = forget_model(
            batch["input_ids"],
            labels=batch["target_ids"],
        ).logits
        forget_outputs = forget_outputs.detach()

    forced_logits = pretrained_outputs - alpha * F.relu(forget_outputs - pretrained_outputs)

    prob_p = torch.nn.functional.softmax(forced_logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs, -1)

    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    return loss



def compute_forced_kl_2(pretrained_model, forget_model, current_model, batch, alpha, T):
    normal_outputs = current_model(
        batch["input_ids"],
        labels=batch["target_ids"],
    ).logits

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"],
            labels=batch["target_ids"],
        ).logits
        pretrained_outputs = pretrained_outputs.detach()
        forget_outputs = forget_model(
            batch["input_ids"],
            labels=batch["target_ids"],
        ).logits
        forget_outputs = forget_outputs.detach()

    forced_logits = pretrained_outputs - alpha * F.relu(forget_outputs - pretrained_outputs)
    
    prob_p = F.log_softmax(forced_logits/T, -1)
    prob_q = F.softmax(normal_outputs/T, -1)
    loss = F.kl_div(prob_p, prob_q, size_average=False) * (T**2) / forced_logits.shape[0]

    return loss


def get_answer_loss( batch, model):
    input_ids, labels = (
        batch["input_ids"],
        batch["target_ids"],
    )
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    return loss



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def evaluate_ndcg(user2item_test, user2items_top, top_k):
    dcgs = [1 / math.log2(i + 2) for i in range(top_k)]
    ndcg = 0
    for u, items in user2items_top.items():
        ground_truth = set(user2item_test[u])
        dcg = 0
        count = 0
        for idx, item in enumerate(items[:top_k]):
            if item in ground_truth:
                dcg += dcgs[idx]
                count += 1
        if count > 0:
            dcg = dcg / sum(dcgs[:count])
        ndcg += dcg
    return ndcg / len(user2item_test)


def evaluate_hr(user2item_test, user2items_top, top_k):
    total = 0
    for u, items in user2items_top.items():
        ground_truth = set(user2item_test[u])
        count = 0
        for item in items[:top_k]:
            if item in ground_truth:
                count += 1
        total += count / len(ground_truth)

    return total / len(user2item_test)


def ids2tokens(ids, tokenizer):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return text.split()