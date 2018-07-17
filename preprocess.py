import glob
import os
import json
from nltk.tokenize import sent_tokenize, word_tokenize

def sent_word_tokenize(text):
    sents = sent_tokenize(text)
    words = [word_tokenize(s) for s in sents]
    return words

def preprocess(task):
    print('Preprocessing the dataset ' + task + '...')
    dataset_names = ['train', 'dev', 'test']
    level_names = ['high', 'middle']
    q_id = 0
    label_dict = {'A':0, 'B':1, 'C':2, 'D':3}
    for dataset_name in dataset_names:
        data_all = []
        for level_name in level_names:
            path = os.path.join('data', task, 'RACE', dataset_name, level_name)
            filenames = glob.glob(path+'/*txt')
            for filename in filenames:
                with open(filename, 'r', encoding='utf-8') as fpr:
                    data_raw = json.load(fpr)
                    artical = [ word_tokenize(s.strip()) for s in sent_tokenize(data_raw['article']) ]
                    for i in range(len(data_raw['answers'])):
                        instance = {}
                        instance['ground_truth'] = label_dict[ data_raw['answers'][i] ]
                        instance['options'] = [ word_tokenize( option ) for option in data_raw['options'][i] ]
                        instance['question'] = word_tokenize( data_raw['questions'][i] )
                        instance['article'] = artical
                        instance['q_id'] = q_id
                        instance['filename'] = filename
                        q_id += 1
                        data_all.append(instance)
                        if len(data_all) % 1000==0:
                            print(len(data_all))
        with open(os.path.join('data', task, 'sequence', dataset_name)+'.json', 'w', encoding='utf-8') as fpw:
            json.dump(data_all, fpw)

if __name__ == '__main__':
    preprocess('race')
