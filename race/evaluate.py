'''
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.
Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.
This software is provided by the copyright holder and creator "as is" and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
'''
import torch

def accuracy(ground_truth, prediction):
    assert(len(ground_truth) == len(prediction))
    accuracy = float( (ground_truth==prediction).float().mean(0) )
    return accuracy

def evaluation(model, optimizer, criterion, corpus, cuda, batch_size, dataset='dev'):
    model.eval()
    labels_all = []
    pred_all = []
    total_loss = 0
    count = 0
    while True:
        data = corpus.get_batch(batch_size, dataset)
        output = model(data)
        _, pred = output.max(1)
        pred_all.append(pred.cpu())
        labels_all.append(data[3])
        labels = data[3].cuda() if cuda else data[3]
        loss = criterion(output, labels)

        loss = loss.detach()
        total_loss += float(loss * output.size(0))
        count += output.size(0)

        if corpus.start_id[dataset] >= len(corpus.data_all[dataset]): break


    loss = total_loss / count
    score = accuracy( torch.cat(labels_all), torch.cat(pred_all) )

    model.train()
    return score
