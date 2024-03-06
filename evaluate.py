import os
import torch
import argparse
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
from accelerate import Accelerator
from utils import  now_time, str2bool, get_loader
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def main(args):
    accelerator = Accelerator()
    print(now_time() + 'Loading data')
    device = accelerator.device

    with open(args.teacher_path, 'rb') as f:
        teacher_model = torch.load(f)
    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f)
        
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)

    test_loader = get_loader('test', args.data_dir+args.test_data, tokenizer, args.batch_size)
    forget_loader = get_loader('test', args.data_dir + args.forget_data, tokenizer, args.batch_size)

    teacher_model, model, test_loader, forget_loader = accelerator.prepare(teacher_model, model, test_loader, forget_loader)
    
    print(now_time() + 'test')
    evaluate(teacher_model, model, test_loader, device, accelerator)
    print(now_time() + 'forget')
    evaluate(teacher_model, model, forget_loader, device, accelerator)


def evaluate(teacher_model, model, loader, device, accelerator):
    model.eval()
    teacher_model.eval()

    pred_list, label_list = [], []
    loss_list = []
    l2_list = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids']
            lm_labels = batch["target_ids"]

            p_outputs = teacher_model(input_ids=input_ids, labels=lm_labels).logits
            q_outputs = model(input_ids=input_ids, labels=lm_labels).logits

            prob_p = torch.nn.functional.softmax(p_outputs[:, 0, :], -1)
            prob_q = torch.nn.functional.softmax(q_outputs[:, 0, :], -1)
            loss = 0.5* (prob_p * torch.log( (prob_p + 1e-12) / (prob_q + 1e-12))).sum(-1) + 0.5* (prob_q * torch.log( (prob_q + 1e-12) / (prob_p + 1e-12))).sum(-1)
            loss = loss.contiguous()
            loss_list.append( accelerator.gather_for_metrics(loss).cpu().numpy())

            l2_norm = torch.norm(prob_p-prob_q, p=2, dim=-1)
            # l2_norm = torch.norm(p_outputs[:, 0, :]-q_outputs[:, 0, :], 2)
            l2_norm = l2_norm.contiguous()
            l2_list.append(accelerator.gather_for_metrics(l2_norm).cpu().numpy())

            logits = q_outputs
            labels_index = torch.argwhere(torch.bitwise_or(lm_labels == 2163, lm_labels == 465))
            gold = torch.where(lm_labels[labels_index[:, 0], labels_index[:, 1]] == 465, 0, 1)

            logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [465, 2163]]
            prob = torch.softmax(logits, dim=-1)
            pred = prob[:, 1]
            pred = pred.contiguous()
            gold = gold.contiguous()

            pred_list.append( accelerator.gather_for_metrics(pred).cpu().numpy())
            label_list.append( accelerator.gather_for_metrics(gold).cpu().numpy())

    pred = np.concatenate(pred_list)
    gold = np.concatenate(label_list)

    confi = [ data if gold[idx] > 0.5 else (1-data) for idx, data in enumerate(pred)  ]
    confi = np.array(confi).mean()

    auc = roc_auc_score(gold, pred)
    ll = log_loss(gold, pred)
    acc = accuracy_score(gold, pred > 0.5)
    jsd = np.mean(np.concatenate(loss_list))
    l2_dis = np.mean(np.concatenate(l2_list))

    print("AUC,LL,ACC,Confi,JSD,L2: ", auc,ll,acc, confi,jsd,l2_dis)
    return auc, ll ,acc



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model_dir', type=str, default='pretrained_models/t5-base')
  
    parser.add_argument('--data_dir', type=str, default='./datasets/ml-1m/benchmark_proc_data/data/')
    
    parser.add_argument('--test_data', type=str, default='test/test_10_simple.json')
    parser.add_argument('--forget_data', type=str, default='train/forget_0.1_user_10_simple.json')

    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='report interval')
    
    parser.add_argument('--teacher_path', type=str, default='./',
                        help='directory to save the final model')
    parser.add_argument('--checkpoint', type=str, default='./',
                        help='directory to save the final model')


    args = parser.parse_args()

    print('-' * 40 + 'ARGUMENTS' + '-' * 40)
    for arg in vars(args):
        print('{:40} {}'.format(arg, getattr(args, arg)))
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)

    main(args)



