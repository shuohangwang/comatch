# comatch
Implementation of the model described in the ACL short paper:
- [A Co-Matching Model for Multi-choice Reading Comprehension](https://arxiv.org/abs/1806.04068) by Shuohang Wang, Mo Yu, Shiyu Chang, Jing Jiang

### Requirements
- Pytorch 0.4
- Python 3.6
- NLTK

### Datasets
- [RACE: Large-scale ReAding Comprehension Dataset From Examinations](http://www.cs.cmu.edu/~glai1/data/race/)
- [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/data/glove.840B.300d.zip)

### Usage
```
sh preprocess.sh
python main.lua --cuda
