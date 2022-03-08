from torch import nn
from torch.nn import functional as F
import torch

class PV_DM(nn.Module):
    def __init__(self, vocab_size,doc_size, embd_size, context_size, hidden_size):
        super(PV_DM, self).__init__()
        self.doc_embeddings = nn.Embedding(doc_size,embd_size)
        self.word_embeddings = nn.Embedding(vocab_size, embd_size)
        self.linear1 = nn.Linear((2 * context_size+1) * embd_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, doc_ids,context_ids):
        batch_size = context_ids.shape[0]
        doc_embedded = self.doc_embeddings(doc_ids)
        word_embedded = self.word_embeddings(context_ids)
        embedded = torch.cat((doc_embedded,word_embedded),dim=1).view(batch_size,-1)
        hid = F.relu(self.linear1(embedded))
        out = self.linear2(hid)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

class PV_DM_NegSample(nn.Module):
    def __init__(self, vocab_size,doc_size, embd_size):
        super(PV_DM_NegSample, self).__init__()
        self.doc_embeddings = nn.Embedding(doc_size, embd_size)
        self.word_embeddings = nn.Embedding(vocab_size, embd_size)
        self.out = nn.Embedding(vocab_size,embd_size)

    def forward(self,doc_ids,context_ids, target_noise_ids):
        doc_embedded = self.doc_embeddings(doc_ids)
        word_embedded = self.word_embeddings(context_ids)
        target_noise_embedded = self.word_embeddings(target_noise_ids)
        embedded = torch.mean(torch.cat((doc_embedded, word_embedded), dim=1),1,True)
        return torch.bmm(embedded, target_noise_embedded.permute(0,2,1)).squeeze()
