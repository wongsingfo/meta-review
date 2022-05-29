
import logging
import json
import os
import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from parser_util import get_parser

from datasets import load_dataset
from torch.utils.data import DataLoader
from losses import Loss_fn
from transformers import AdamW, get_linear_schedule_with_warmup
from encoder import MetaModel
from collections import defaultdict

def load_data(args):

    train_file = args.dataPath + "train_rate_concat_seg-ctrl.csv"
    validation_file = args.dataPath + "val_rate_concat_seg-ctrl.csv"
    test_file = args.dataPath + "test_rate_concat_seg-ctrl.csv"

    data_files = {}
    if train_file is not None:
        data_files["train"] = train_file
        extension = train_file.split(".")[-1]
    if validation_file is not None:
        data_files["validation"] = validation_file
        extension = validation_file.split(".")[-1]
    if test_file is not None:
        data_files["test"] = test_file
        extension = test_file.split(".")[-1]
    datasets = load_dataset(extension, data_files=data_files)
    # load datas

    column_names = datasets["train"].column_names
    # print(column_names)
    # Get the column names for input/target.
    text_column = column_names[0]
    summary_column = column_names[1]
    
    # print(text_column)
    # print(summary_column) 
      

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    predict_dataset = datasets["test"]

   
    tr_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(eval_dataset, batch_size=args.test_batch_size, shuffle=True)
    test_dataloader = DataLoader(predict_dataset, batch_size=args.test_batch_size, shuffle=False)



    return tr_dataloader, val_dataloader, test_dataloader


def init_model(args):
    device = torch.device('cuda', args.numDevice)
    torch.cuda.set_device(device)
    model = MetaModel().to(device)
    return model


def init_optim(args, model):

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    return optimizer

def init_lr_scheduler(args, optim):
    '''
    Initialize the learning rate scheduler
    '''
    
    t_total = args.epochs * int(5355/args.train_batch_size)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    return scheduler

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)



all_labels = ['abstract', 'strength', 'weakness', 'rating_summary', 'ac_disagreement', 'rebuttal_process', 'suggestion', 'decision', 'misc']


def deal_data(labels):
    label_ids = []

    for line in labels:
        line = line.split(' | ')
        tmp = []
        for l in all_labels:
            if l in line:
                tmp.append(1)
            else:
                tmp.append(0)
        label_ids.append(tmp)
    return label_ids

def train(args, tr_dataloader, model, optim, lr_scheduler, val_dataloader):

  
    train_loss, epoch_train_loss = [], []
    train_acc, epoch_train_acc = [], []
    val_loss, epoch_val_loss = [], []
    val_acc, epoch_val_acc = [], []
    best_acc = 0
    
    loss_fn = Loss_fn(args)

    acc_best_model_path = os.path.join(args.fileModelSave, 'acc_best_model.pth')
    

    for epoch in range(args.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        model.train()
        
        for  i, batch in enumerate(tr_dataloader):
            optim.zero_grad()
            text, labels = batch['text'], batch['summary']

            labelids = deal_data(labels)
           
            model_outputs = model(text)
            # print(re)
            
            loss, p, r, f, acc, _ = loss_fn(model_outputs, labelids)
            # print(labels)
            loss.backward()
            optim.step()
            lr_scheduler.step()
        
            train_loss.append(loss.item())
            # train_p.append(p)
            # train_r.append(r)
            # train_f1.append(f)

            train_acc.append(acc)
            print('Batch: {}, Train Loss: {}, Train p: {}, Train r: {}, Train f1: {} Train acc: {}'.format(i, loss, p, r, f, acc))


        avg_loss = np.mean(train_loss[-i:])
        avg_acc = np.mean(train_acc[-i:])
                

        print('Avg Train Loss: {},  Avg Train acc: {}'.format(avg_loss, avg_acc))
        epoch_train_loss.append(avg_loss)
        epoch_train_acc.append(avg_acc)

       
        model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_dataloader)):
                text, labels = batch['text'], batch['summary']
                labelids = deal_data(labels)
                model_outputs = model(text)
                loss, p, r, f, acc, _ = loss_fn(model_outputs, labelids)

                val_loss.append(loss.item())
                val_acc.append(acc)
                
            
        avg_loss = np.mean(val_loss[-i:])
        avg_acc = np.mean(val_acc[-i:])
        

        # print('Avg Train Loss: {}, Avg Train Precision: {}, Avg Train Recall: {},Avg Train F1: {}'.format(avg_loss, avg_p, avg_r, avg_f1))
        epoch_val_loss.append(avg_loss)
        epoch_val_acc.append(avg_acc)
        
        
        acc_prefix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
       
        print('Avg Val Loss: {}, Avg Val acc: {}{}'.format(avg_loss, avg_acc, acc_prefix))
   
        
        torch.save(model.state_dict(), acc_best_model_path)
        best_acc = avg_acc
        acc_best_state = model.state_dict()
        
       

    # torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss',  'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(args.fileModelSave,
                                       name + '.txt'), locals()[name])

    return acc_best_state
        

def test(args, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    # write predicts
    id2label = defaultdict(str)
    for i in range(len(all_labels)):
        id2label[i] = all_labels[i]
    print(id2label)
    with torch.no_grad():
        
        val_loss = []
        val_acc = []
        loss_fn = Loss_fn(args)
        model.eval()
        predict = []

        for i, batch in tqdm(enumerate(test_dataloader)):
            text, labels = batch['text'], batch['summary']
            labelids = deal_data(labels)
            model_outputs = model(text)
            # print(re)
            loss, p, r, f, acc, pred = loss_fn(model_outputs, labelids)
            predict.append(pred)

            val_loss.append(loss.item())
            val_acc.append(acc)
            # if i >= 50:
            #     break
            # break
            # print(pred)
            # print(pred.shape)
            # print(re)
                           
        avg_loss = np.mean(val_loss)
        avg_acc = np.mean(val_acc)
        predict = torch.stack(predict)
        predict = predict.view(-1, 9)
        predict = predict.cpu().detach()
        
       
        print('Test acc: {}'.format(avg_acc))
        print('Test Loss: {}'.format(avg_loss))
        path = args.fileModelSave + "/test_score.json"
        with open(path, "w") as fout:
            tmp = {"acc": avg_acc, "Loss": avg_loss}
            fout.write("%s\n" % json.dumps(tmp, ensure_ascii=False))
       
        
        
        path = args.fileModelSave + "/predictions.json"
        with open(path, "w") as fout:
            for line in predict:
                print(line)
                tmp = ""
                for j, item in enumerate(line):
                    if item == 1:
                        tmp += (" | " + id2label[j])
                tmp = tmp[3:]
                fout.write("%s\n" % tmp)



def main():
    args = get_parser().parse_args()
    if not os.path.exists(args.fileModelSave):
        os.makedirs(args.fileModelSave)

    tr_dataloader, val_dataloader, test_dataloader = load_data(args)
    
    model = init_model(args)


    optim = init_optim(args, model)
    lr_scheduler = init_lr_scheduler(args, optim)
    results = train(args=args,
                    tr_dataloader=tr_dataloader,
                    val_dataloader=val_dataloader,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler)
    

    # print('Testing with last model..')
    # test(args=args,
    #      test_dataloader=test_dataloader,
    #      model=model)

    model.load_state_dict(torch.load(args.fileModelSave + "/acc_best_model.pth"))
    print('Testing with acc best model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model)

    

    return 




if __name__ == "__main__":
    main()
