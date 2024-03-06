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
from utils import (
    compute_kl,
    get_answer_loss,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    
    # unlearning 之后保存的路径
    if accelerator.is_main_process:
        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)

    
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)

    forget_loader = get_loader('train', args.data_dir + args.forget_data, tokenizer, args.batch_size)

    test_forget_loader = get_loader('test', args.data_dir + args.forget_data, tokenizer, args.batch_size)
    valid_loader = get_loader('valid', args.data_dir+'valid/valid_10_simple.json', tokenizer, args.batch_size)
    test_loader = get_loader('test', args.data_dir+'test/test_10_simple.json', tokenizer, args.batch_size)
    
    # 从之前model出发finetune新模型
    with open(os.path.join(args.language_model_path, 'model.pt'), 'rb') as f:
        model = torch.load(f)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, optimizer, forget_loader, valid_loader, test_loader, test_forget_loader = accelerator.prepare(
        model, optimizer, forget_loader, valid_loader,test_loader, test_forget_loader)
    
    accelerator.print(now_time() + 'Start training')
    best_val_loss = float('inf')
    endure_count = 0
    #记录总步数
    step = 0

    for epoch in range(1, args.epochs + 1):
        accelerator.print(now_time() + 'epoch {}'.format(epoch))
        model.train()
        
        accelerator.wait_for_everyone()

        # retain 训练
        for forget_batch in forget_loader:
            step += 1
            model.train()
            optimizer.zero_grad()

            loss = get_answer_loss(forget_batch, model)
            
            accelerator.backward(loss)
            optimizer.step()

            if step % args.log_interval == 0 :
                print(now_time() + 'Step {:5d} Loss {:4.4f}'.format(step, loss))

                accelerator.wait_for_everyone()

                accelerator.print(now_time() + 'validation')

                loss, auc,ll,acc = evaluate(model, test_loader, device, accelerator)
                accelerator.print("Test :Loss, AUC, LL, ACC: ", loss, auc,ll,acc)
                loss, auc,ll,acc = evaluate(model, test_forget_loader, device, accelerator)
                accelerator.print("Forget :Loss, AUC, LL, ACC: ", loss, auc,ll,acc)
                
                accelerator.print("save model")
                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    with open(os.path.join(args.checkpoint, f'model_{step}.pt'), 'wb') as f:
                        torch.save(unwrapped_model, f)


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
    accelerator.print(gold.shape)
    
    auc = roc_auc_score(gold, pred)
    ll = log_loss(gold, pred)
    acc = accuracy_score(gold, pred > 0.5)
    return ret_loss, auc, ll ,acc



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model_dir', type=str, default='pretrained_models/t5-base')

    parser.add_argument('--data_dir', type=str, default='datasets/ml-1m/benchmark_proc_data/data/')

    # parser.add_argument('--train_data', type=str, default='train/retain_0.1_user_10_simple.json')
    parser.add_argument('--forget_data', type=str, default='train/forget_0.1_user_10_simple.json')
    
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
 
    parser.add_argument('--log_interval', type=int, default=200,
                        help='report interval')
    
    parser.add_argument('--checkpoint', type=str, default='./',
                        help='directory to save the final model')
    parser.add_argument('--language_model_path', type=str, default='checkpoint/ml-1m-base-original-0.0005/model.pt')
    
    parser.add_argument('--endure_times', type=int, default=3,
                        help='the maximum endure times of loss increasing on validation')

    args = parser.parse_args()

    print('-' * 40 + 'ARGUMENTS' + '-' * 40)
    for arg in vars(args):
        print('{:40} {}'.format(arg, getattr(args, arg)))
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)

    main(args)



