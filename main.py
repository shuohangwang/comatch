# coding: utf-8
'''
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.
Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.
This software is provided by the copyright holder and creator "as is" and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
'''
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from utils.corpus import Corpus
from race.comatch import CoMatch
from race.evaluate import evaluation
import json

parser = argparse.ArgumentParser(description='Multiple Choice Reading Comprehension')
parser.add_argument('--task', type=str, default='race',
                    help='task name')
parser.add_argument('--model', type=str, default='CoMatch',
                    help='model name')
parser.add_argument('--emb_dim', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--mem_dim', type=int, default=150,
                    help='hidden memory size')
parser.add_argument('--lr', type=float, default=0.002,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='batch size')
parser.add_argument('--dropoutP', type=float, default=0.2,
                    help='dropout ratio')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--exp_idx', type=str,  default='1',
                    help='experiment index')
parser.add_argument('--log', type=str,  default='nothing',
                    help='take note')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


corpus = Corpus(args.task)
model = eval(args.model)(corpus, args)
model.train()
criterion = nn.NLLLoss()

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adamax(parameters, lr=args.lr)


if args.cuda:
    model.cuda()
    criterion.cuda()

start_time = time.time()
total_loss = 0
interval = args.interval
save_interval = len(corpus.data_all['train']) // args.batch_size

best_dev_score = -99999
iterations = args.epochs*len(corpus.data_all['train']) // args.batch_size
print('max iterations: '+str(iterations))
for iter in range(iterations):
    optimizer.zero_grad()
    data = corpus.get_batch(args.batch_size, 'train')
    output = model(data)
    labels = data[3].cuda() if args.cuda else data[3]
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    total_loss += float(loss.data)

    if iter % interval == 0:
        cur_loss = total_loss / interval if iter!=0 else total_loss
        elapsed = time.time() - start_time
        print('| iterations {:3d} | start_id {:3d} | ms/batch {:5.2f} | loss {:5.3f}'.format(
        iter, corpus.start_id['train'], elapsed * 1000 / interval, cur_loss))
        total_loss = 0
        start_time = time.time()

    if iter % save_interval == 0:

        torch.save([model, optimizer, criterion], 'trainedmodel/'+args.task+'_save.pt')
        score = evaluation(model, optimizer, criterion, corpus, args.cuda, args.batch_size)
        print('DEV accuracy: ' + str(score))

        with open('trainedmodel/'+args.task+'_record.txt', 'a', encoding='utf-8') as fpw:
            if iter == 0: fpw.write(str(args) + '\n')
            fpw.write(str(iter) + ':\tDEV accuracy:\t' + str(score) + '\n')

        if score > best_dev_score:
            best_dev_score = score
            torch.save([model, optimizer, criterion], 'trainedmodel/'+args.task+'_save_best.pt')

    if (iter+1) % (len(corpus.data_all['train']) // args.batch_size) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.95


model, optimizer, criterion = torch.load('trainedmodel/'+args.task+'_save_best.pt')
score = evaluation(model, optimizer, criterion, corpus, args.cuda, args.batch_size, dataset='test')
with open('trainedmodel/'+args.task+'_record.txt', 'a', encoding='utf-8') as fpw:
    fpw.write('TEST accuracy:\t' + str(score) + '\n')
print('TEST accuracy: ' + str(score))
