from torch.utils.data import Dataset
import torch
from gensim.corpora.dictionary import Dictionary

class Doc2VecText(Dataset):
    def __init__(self, tokenized_texts, word_vocab, doc_vocab, window_size):
        """
        :param tokenized_texts: Tokenized text.
        :type tokenized_texts: list(list(list(str),list(str),list(str)))
        tokenized_texts example:
        [[[doc1],[token1,token2,token3],[target]],
        [[doc2],[token2,token5,token6],[target]]]
        Negative Sample tokenized_texts example:
        [[[doc1],[token1,token2,token3],[target1,negasample1,,negasample2]],
        [[doc2],[token2,token5,token6],[target2,negasample3,,negasample4]]]
        """
        self.sents = tokenized_texts
        self._len = len(self.sents)

        self.word_vocab = word_vocab
        self.doc_vocab = doc_vocab

        self.word_size = len(self.word_vocab)
        self.doc_size = len(self.doc_vocab)

        self.window_size = window_size

    def __getitem__(self, index):
        """
        The primary entry point for PyTorch datasets.
        This is were you access the specific data row you want.

        :param index: Index to the data point.
        :type index: int
        """
        doc = self.sents[index][0]
        sent = self.sents[index][1]
        target = self.sents[index][2]
        vectorized_doc = self.doc_vectorize(doc)
        vectorized_sent = self.word_vectorize(sent)
        vectorized_target = self.word_vectorize(target)
        return {'doc':vectorized_doc,
                'context':vectorized_sent,
                'target':vectorized_target}

    def __len__(self):
        return self._len

    def word_vectorize(self, tokens):
        """
        :param tokens: Tokens that should be vectorized.
        :type tokens: list(str)
        """
        return torch.tensor(self.word_vocab.doc2idx(tokens, unknown_word_index=1))

    def word_unvectorize(self, indices):
        """
        :param indices: Converts the indices back to tokens.
        :type tokens: list(int)
        """
        return [self.word_vocab[i] for i in indices]

    def doc_vectorize(self, tokens):
        """
        :param tokens: Tokens that should be vectorized.
        :type tokens: list(str)
        """
        return torch.tensor(self.doc_vocab.doc2idx(tokens))

    def doc_unvectorize(self, indices):
        """
        :param indices: Converts the indices back to tokens.
        :type tokens: list(int)
        """
        return [self.doc_vocab[i] for i in indices]

class Doc2VecText_Pretrain(Dataset):
    def __init__(self, tokenized_texts, word_vocab, doc_vocab, window_size):
        """
        :param tokenized_texts: Tokenized text.
        :type tokenized_texts: list(list(list(str),list(str),list(str)))
        tokenized_texts example:
        [[[doc1],[token1,token2,token3],[target]],
        [[doc2],[token2,token5,token6],[target]]]
        Negative Sample tokenized_texts example:
        [[[doc1],[token1,token2,token3],[target1,negasample1,,negasample2]],
        [[doc2],[token2,token5,token6],[target2,negasample3,,negasample4]]]
        """
        self.sents = tokenized_texts
        self._len = len(self.sents)

        self.word_vocab = self.load_vocab(word_vocab)
        self.doc_vocab = self.load_vocab(doc_vocab)

        self.word_size = len(self.word_vocab)
        self.doc_size = len(self.doc_vocab)

        self.window_size = window_size

    def __getitem__(self, index):
        """
        The primary entry point for PyTorch datasets.
        This is were you access the specific data row you want.

        :param index: Index to the data point.
        :type index: int
        """
        doc = self.sents[index][0]
        sent = self.sents[index][1]
        target = self.sents[index][2]
        vectorized_doc = self.doc_vectorize(doc)
        vectorized_sent = self.word_vectorize(sent)
        vectorized_target = self.word_vectorize(target)
        return {'doc':vectorized_doc,
                'context':vectorized_sent,
                'target':vectorized_target}

    def __len__(self):
        return self._len

    def load_vocab(self,file):
        with open(file) as fin:
            pretrained_keys = {line.strip(): i for i, line in enumerate(fin)}
        vocab = Dictionary({})
        vocab.token2id = pretrained_keys
        return vocab

    def word_vectorize(self, tokens):
        """
        :param tokens: Tokens that should be vectorized.
        :type tokens: list(str)
        """
        return torch.tensor(self.word_vocab.doc2idx(tokens, unknown_word_index=1))

    def word_unvectorize(self, indices):
        """
        :param indices: Converts the indices back to tokens.
        :type tokens: list(int)
        """
        return [self.word_vocab[i] for i in indices]

    def doc_vectorize(self, tokens):
        """
        :param tokens: Tokens that should be vectorized.
        :type tokens: list(str)
        """
        return torch.tensor(self.doc_vocab.doc2idx(tokens))

    def doc_unvectorize(self, indices):
        """
        :param indices: Converts the indices back to tokens.
        :type tokens: list(int)
        """
        return [self.doc_vocab[i] for i in indices]