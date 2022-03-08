import pandas as pd
import jieba
from tqdm.auto import tqdm
from gensim.corpora.dictionary import Dictionary
from nltk import word_tokenize
import numpy as np
from langdetect import detect
jieba.initialize()

class Process_corpus(object):
    def __init__(self,window_size=2,isNegSample=False):
        self.window_size = window_size
        self.isNegSample = isNegSample
        self.stopwords = set([line.strip() for line in open('./config/baidu_stopwords.txt',mode='r',encoding='utf-8')])

    def read_corpus(self,file):
        df = pd.read_csv(file, sep='\t',header=None)
        return df

    @staticmethod
    def is_contains_chinese(strs):
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False

    def process_corpus(self, df):
        df = df.values
        n = 2 * self.window_size+1
        data = []
        for line in tqdm(df):
            doc = line[0]
            des = line[1].lower()
            try:
                lan = detect(des)
            except:
                continue
            if lan in ['zh-cn','zh-tw']:
                tokens = jieba.lcut(des)
            elif lan == 'en':
                tokens = word_tokenize(des)
            else:
                continue
            tokens = [token for token in tokens if token not in self.stopwords]
            if len(tokens) <= n:
                continue
            data.append([[doc],tokens])
        return data

    def build_vocab(self,data):
        word_vocab = Dictionary([sent[1] for sent in data])
        try:
            word_special_tokens = {'<pad>': 0, '<unk>': 1}
            word_vocab.patch_with_special_tokens(word_special_tokens)
        except:
            pass
        doc_vocab = Dictionary([sent[0] for sent in data])

        return word_vocab, doc_vocab
    
    def DM_window(self,data):
        dataset = []
        n = self.window_size * 2 + 1
        for line in tqdm(data):
            doc = line[0]
            tokens = line[1]
            for i, window in enumerate(self.per_window(tokens, n)):
                target = [window.pop()]
                dataset.append([doc,window,target])
        return dataset

    def DM_window_negsample(self,data,sample_size=None):
        dataset = []
        n = self.window_size * 2 + 1
        if not sample_size or sample_size <= 0:
            sample_size = n
        for line in tqdm(data):
            doc = line[0]
            tokens = line[1]
            for i, window in enumerate(self.per_window(tokens, n)):
                target = [window.pop()]
                leftovers = tokens[:i] + tokens[i + n:]
                if target[0] in leftovers:
                    leftovers.remove(target[0])
                negsample = np.random.choice(leftovers,size=sample_size,replace=False).tolist()
                target.extend(negsample)
                dataset.append([doc,window,target])
        return dataset

    @staticmethod
    def per_window(sequence, n=1):
        """
        From http://stackoverflow.com/q/42220614/610569
            >>> list(per_window([1,2,3,4], n=2))
            [(1, 2), (2, 3), (3, 4)]
            >>> list(per_window([1,2,3,4], n=3))
            [(1, 2, 3), (2, 3, 4)]
        """
        start, stop = 0, n
        seq = list(sequence)
        while stop <= len(seq):
            yield seq[start:stop]
            start += 1
            stop += 1

    def preprocess(self,file,sample_size=None):
        df = self.read_corpus(file)
        print('start process corpus')
        data = self.process_corpus(df)
        word_vocab, doc_vocab = self.build_vocab(data)
        print('start build DM windows')
        if self.isNegSample:
            dataset = self.DM_window_negsample(data,sample_size)
        else:
            dataset = self.DM_window(data)
        print(f'remain {len(dataset)} rows')
        print(f'word vocab size: {len(word_vocab)}')
        print(f'doc vocab size: {len(doc_vocab)}')
        return dataset, word_vocab, doc_vocab

    __call__ = preprocess
