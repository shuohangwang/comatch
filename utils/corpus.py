'''
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.
Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.
This software is provided by the copyright holder and creator "as is" and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
'''
import os
import torch
import glob
import json
import torch

def prep_glove():
    vocab = {}
    ivocab = []
    tensors = []
    with open('data/glove/glove.840B.300d.txt', 'r', encoding='utf8') as f:
        for line in f:
            vals = line.rstrip().split(' ')
            if len(vals) != 301:
                print(line)
                continue
            assert(len(vals) == 301)
            word = vals[0]
            vec = torch.FloatTensor([ float(v) for v in vals[1:] ])
            vocab[word] = len(ivocab)
            ivocab.append(word)
            tensors.append(vec)
            assert (vec.size(0) == 300)
    assert len(tensors) == len(ivocab)
    tensors = torch.cat(tensors).view(len(ivocab), 300)
    with open('data/glove/glove_emb.pt', 'wb') as fpw:
        torch.save([tensors, vocab, ivocab], fpw)


class Dictionary(object):
    def __init__(self, task):
        self.task = task
        filename = os.path.join('data', self.task, 'word2idx.pt')

        if os.path.exists(filename):
            self.word2idx = torch.load(os.path.join('data', self.task, 'word2idx.pt'))
            self.idx2word = torch.load(os.path.join('data', self.task, 'idx2word.pt'))
            self.word2idx_count = torch.load(os.path.join('data', self.task, 'word2idx_count.pt'))
        else:
            self.word2idx = {'<<padding>>':0, '<<unk>>':1}
            self.word2idx_count = {'<<padding>>':0, '<<unk>>':0}

            self.idx2word = ['<<padding>>', '<<unk>>']

            self.build_dict('train')
            self.build_dict('dev')
            if self.task != 'squad':
                self.build_dict('test')

            torch.save(self.word2idx, os.path.join('data', self.task, 'word2idx.pt'))
            torch.save(self.idx2word, os.path.join('data', self.task, 'idx2word.pt'))
            torch.save(self.word2idx_count, os.path.join('data', self.task, 'word2idx_count.pt'))

        filename_emb = os.path.join('data', task, 'embeddings.pt')
        if os.path.exists(filename_emb):
            self.embs = torch.load(filename_emb)
        else:
            self.embs = self.build_emb()

        print ("vacabulary size: " + str(len(self.idx2word)))

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)

            self.word2idx_count[word] = 1
        else:
            self.word2idx_count[word] += 1

        return self.word2idx[word]

    def build_dict(self, dataset):
        filename = os.path.join('data', self.task, 'sequence', dataset+'.json')

        if self.task == 'race':
            with open(filename, 'r', encoding='utf-8') as fpr:
                data_all = json.load(fpr)
                for instance in data_all:
                    words = instance['question']
                    for option in instance['options']: words += option
                    for sent in instance['article']: words += sent
                    for word in words: self.add_word(word)
        else:
            assert False, 'the task ' + self.task + ' is not supported!'

    def build_emb(self, all_vacob=False, filter=False, threshold=10):
        word2idx = torch.load(os.path.join('data', self.task, 'word2idx.pt'))
        idx2word = torch.load(os.path.join('data', self.task, 'idx2word.pt'))
        emb = torch.FloatTensor(len(idx2word), 300).zero_()
        print ("Loading Glove ...")
        print ("Raw vacabulary size: " + str(len(idx2word)) )

        if not os.path.exists('data/glove/glove_emb.pt'): prep_glove()
        glove_tensors, glove_vocab, glove_ivocab = torch.load('data/glove/glove_emb.pt')

        if not all_vacob:
            self.word2idx = {'<<padding>>':0, '<<unk>>':1}
            self.idx2word = ['<<padding>>', '<<unk>>']
        count = 0
        for w_id, word in enumerate(idx2word):
            if word in glove_vocab:
                id = self.add_word(word)
                emb[id] = glove_tensors[glove_vocab[word]]
                count += 1
        emb = emb[:len(self.idx2word)]

        print("Number of words not appear in glove: " + str(len(idx2word)-count) )
        print ("Vacabulary size: " + str(len(self.idx2word)))
        torch.save(emb, os.path.join('data', self.task, 'embeddings.pt'))
        torch.save(self.word2idx, os.path.join('data', self.task, 'word2idx.pt'))
        torch.save(self.idx2word, os.path.join('data', self.task, 'idx2word.pt'))

        return emb

    def filter(self, threshold=10):
        for word, count in self.word2idx_count.items():
            if count > threshold and word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, task):
        self.task = task
        self.dictionary = Dictionary(task)

        self.data_all, self.start_id, self.indices = {}, {}, {}
        setnames = ['train', 'dev', 'test']
        for setname in setnames:
            self.data_all[setname] = self.load_data(os.path.join('data', self.task, 'sequence', setname) + '.json')
            print(setname, len(self.data_all[setname]))
            self.start_id[setname] = 0
            self.indices[setname] = torch.randperm(len(self.data_all[setname])) if setname == 'train' else torch.range(0, len(self.data_all[setname])-1)

    def seq2tensor(self, words):
        seq_tensor = torch.LongTensor(len(words))
        for id, word in enumerate(words):
            seq_tensor[id] = self.dictionary.word2idx[word] if word in self.dictionary.word2idx else 1
        return seq_tensor

    def load_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as fpr:
            data = json.load(fpr)
        return data

    def get_batch(self, batch_size, setname):
        if self.start_id[setname] >= len(self.data_all[setname]):
            self.start_id[setname] = 0
            if setname == 'train': self.indices[setname] = torch.randperm(len(self.data_all[setname]))

        end_id = self.start_id[setname] + batch_size if self.start_id[setname] + batch_size < len(self.data_all[setname]) else len(self.data_all[setname])
        documents, questions, options, labels = [], [], [], []
        for i in range(self.start_id[setname], end_id):
            instance_id = int(self.indices[setname][i])

            instance = self.data_all[setname][instance_id]

            questions.append(instance['question'])
            options.append(instance['options'])
            documents.append(instance['article'])
            labels.append(instance['ground_truth'])

        self.start_id[setname] += batch_size

        questions = self.seq2tensor(questions)
        documents = self.seq2Htensor(documents)
        options = self.seq2Htensor(options)
        labels = torch.LongTensor(labels)
        return [documents, questions, options, labels]

    def seq2tensor(self, sents, sent_len_bound=50):
        sent_len_max = max([len(s) for s in sents])
        sent_len_max = min(sent_len_max, sent_len_bound)

        sent_tensor = torch.LongTensor(len(sents), sent_len_max).zero_()

        sent_len = torch.LongTensor(len(sents)).zero_()
        for s_id, sent in enumerate(sents):
            sent_len[s_id] = len(sent)
            for w_id, word in enumerate(sent):
                if w_id >= sent_len_max: break
                sent_tensor[s_id][w_id] = self.dictionary.word2idx.get(word, 1)
        return [sent_tensor, sent_len]

    def seq2Htensor(self, docs, sent_num_bound=50, sent_len_bound=50):
        sent_num_max = max([len(s) for s in docs])
        sent_num_max = min(sent_num_max, sent_num_bound)
        sent_len_max = max([len(w) for s in docs for w in s ])
        sent_len_max = min(sent_len_max, sent_len_bound)

        sent_tensor = torch.LongTensor(len(docs), sent_num_max, sent_len_max).zero_()
        sent_len = torch.LongTensor(len(docs), sent_num_max).zero_()
        doc_len = torch.LongTensor(len(docs)).zero_()
        for d_id, doc in enumerate(docs):
            doc_len[d_id] = len(doc)
            for s_id, sent in enumerate(doc):
                if s_id >= sent_num_max: break
                sent_len[d_id][s_id] = len(sent)
                for w_id, word in enumerate(sent):
                    if w_id >= sent_len_max: break
                    sent_tensor[d_id][s_id][w_id] = self.dictionary.word2idx.get(word, 1)
        return [sent_tensor, doc_len, sent_len]
