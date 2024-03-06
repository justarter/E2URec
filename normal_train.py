import os
import torch
import argparse
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
from accelerate import Accelerator
from utils import  now_time, str2bool, get_loader
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np
import time

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def main(args):
    accelerator = Accelerator()

    accelerator.print(now_time() + 'Loading data')
    device = accelerator.device

    if accelerator.is_main_process:
        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)
    model_path = os.path.join(args.checkpoint, 'model.pt')
    
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)

    train_loader = get_loader('train', args.data_dir + args.train_data, tokenizer, args.batch_size)

    valid_loader = get_loader('valid', args.data_dir+args.valid_data, tokenizer, args.batch_size)
    test_loader = get_loader('test', args.data_dir+args.test_data, tokenizer, args.batch_size)
    
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, optimizer, train_loader, valid_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader, test_loader)
    
    # val_loss = valid_step(model, valid_loader, device, accelerator)
    # evaluate(model, test_loader, device, accelerator)
    
    accelerator.print(now_time() + 'Start training')
    best_val_loss = float('inf')
    endure_count = 0
    time_list = []

    for epoch in range(1, args.epochs + 1):
        accelerator.print(now_time() + 'epoch {}'.format(epoch))
        model.train()
        text_loss = 0.
        total_sample = 0
        step = 0

        accelerator.wait_for_everyone()

        beg = time.time()
        for batch in train_loader:
            step += 1
            input_ids = batch['input_ids']
            lm_labels = batch["target_ids"]
        
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=lm_labels)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()

            batch_size = input_ids.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if step % args.log_interval == 0 or step % len(train_loader) == 0:
                cur_t_loss = text_loss / total_sample
                print(now_time() + 'text loss {:4.4f} | {:5d}/{:5d} batches'.format(cur_t_loss,step, len(train_loader)))
                text_loss = 0.
                total_sample = 0

        cost_time = time.time() - beg
        time_list.append(cost_time)
        accelerator.print('Cost time {:4.4f}'.format(cost_time))

        accelerator.wait_for_everyone()

        accelerator.print(now_time() + 'validation')
        val_loss = valid_step(model, valid_loader, device, accelerator)
        evaluate(model, test_loader, device, accelerator)
        accelerator.print(now_time() + 'valid loss {:4.4f}'.format(val_loss))

        if val_loss < best_val_loss:
            accelerator.print("save model")
            best_val_loss = val_loss
            endure_count = 0
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                with open(model_path, 'wb') as f:
                    torch.save(unwrapped_model, f)
        else:
            endure_count += 1
            accelerator.print(now_time() + 'Endured {} time(s)'.format(endure_count))
            if endure_count == args.endure_times:
                accelerator.print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                break

    accelerator.print('Total Train Time: ', np.array(time_list[:-args.endure_times]).sum())


def valid_step(model, loader, device, accelerator):
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids']
            lm_labels = batch["target_ids"]
            outputs = model(input_ids=input_ids, labels=lm_labels)
            loss = outputs.loss
            loss = accelerator.reduce(loss, reduction='mean')
            batch_size = input_ids.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size
    # print(total_sample)
    return text_loss / total_sample

def evaluate(model, loader, device, accelerator):
    model.eval()
    text_loss = 0.
    total_sample = 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids']
            lm_labels = batch["target_ids"]
            outputs = model(input_ids=input_ids, labels=lm_labels)

            loss = outputs.loss
            logits = outputs.logits
            # print(lm_labels.shape, logits.shape)
            labels_index = torch.argwhere(torch.bitwise_or(lm_labels == 2163, lm_labels == 465))
            gold = torch.where(lm_labels[labels_index[:, 0], labels_index[:, 1]] == 465, 0, 1)
         
            logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [465, 2163]]
            prob = torch.softmax(logits, dim=-1)
            pred = prob[:, 1]
            pred = pred.contiguous()
            gold = gold.contiguous()

            pred_list.append( accelerator.gather_for_metrics(pred).cpu().numpy())
            label_list.append( accelerator.gather_for_metrics(gold).cpu().numpy())

            batch_size = input_ids.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

    ret_loss = text_loss / total_sample

    pred = np.concatenate(pred_list)
    gold = np.concatenate(label_list)
    # accelerator.print(gold.shape)

    auc = roc_auc_score(gold, pred)
    ll = log_loss(gold, pred)
    acc = accuracy_score(gold, pred > 0.5)
    accelerator.print("AUC,LL,ACC: ", auc,ll,acc)
    return ret_loss, auc, ll ,acc



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model_dir', type=str, default='pretrained_models/t5-base')
    parser.add_argument('--data_dir', type=str, default='./datasets/ml-1m/benchmark_proc_data/data/')

    parser.add_argument('--train_data', type=str, default='train/train_10_simple.json')
    parser.add_argument('--valid_data', type=str, default='valid/valid_10_simple.json')
    parser.add_argument('--test_data', type=str, default='test/test_10_simple.json')
    
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='report interval')
    parser.add_argument('--checkpoint', type=str, default='./pod/',
                        help='directory to save the final model')
    parser.add_argument('--endure_times', type=int, default=3,
                        help='the maximum endure times of loss increasing on validation')

    args = parser.parse_args()

    print('-' * 40 + 'ARGUMENTS' + '-' * 40)
    for arg in vars(args):
        print('{:40} {}'.format(arg, getattr(args, arg)))
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)

    main(args)



