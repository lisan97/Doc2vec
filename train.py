import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from prepare_data import Process_corpus
from data_utils import Doc2VecText
from model import PV_DM, PV_DM_NegSample
from loss import NegativeSampling
from lazyme import color_str
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
import argparse

sns.set_style("darkgrid")
sns.set(rc={'figure.figsize':(12, 8)})

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='./data/doc2vec_sample.csv', help="input directory")
parser.add_argument('--test_dir', default='./data/doc2vec_sample.csv', help="test directory")
parser.add_argument('--embed_size', default=128,help='embedding size', type=int)
parser.add_argument('--window_size', default=2,help='sliding window size', type=int)
parser.add_argument('--batch_size', default=64,help='batch size', type=int)
parser.add_argument('--hidden_size', default=128,help='hidden state size', type=int)
parser.add_argument('--learning_rate', default=0.001,help='learning rate', type=float)
parser.add_argument('--weight_decay', default=0.01,help='weight decay', type=float)
parser.add_argument('--isNegSample', default=False,help='whether do Negative Sample', type=bool)
parser.add_argument('--sample_size', default=-1,help='Negative Sample size',type=int)
parser.add_argument('--num_epochs', default=100,help='number of epochs', type=int)
parser.add_argument('--patience', default=15,help='early stop patience', type=int)

class Trainer(object):
    def __init__(self,file=None,
                 batch_size=64,
                 embed_size=128,
                 window_size=2,
                 hidden_size=128,
                 learning_rate=0.001,
                 weight_decay=0.01,
                 isNegSample=False,
                 sample_size=-1):
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.isNegSample = isNegSample
        self.dataloader, self.model, self.optimizer, self.criterion, self.dataset = self.initialize_data_model_optim_loss(file,sample_size)

    def initialize_data_model_optim_loss(self,file,sample_size):
        # Initialize the dataset and dataloader.
        process_corpus = Process_corpus(window_size=self.window_size,isNegSample=self.isNegSample)
        tokenized_text, word_vocab, doc_vocab = process_corpus(file,sample_size)
        word_vocab.save('./config/word_vocab')
        doc_vocab.save('./config/doc_vocab')
        Doc2Vec_data = Doc2VecText(tokenized_text,word_vocab, doc_vocab,self.window_size)
        dataloader = DataLoader(dataset=Doc2Vec_data,
                                batch_size=self.batch_size,
                                shuffle=True)

        # Loss function.
        if self.isNegSample:
            criterion = NegativeSampling()
        else:
            criterion = nn.NLLLoss(ignore_index=Doc2Vec_data.word_vocab.token2id['<pad>'],
                                          reduction='mean')

        # Model.
        if self.isNegSample:
            model = PV_DM_NegSample(len(Doc2Vec_data.word_vocab),len(Doc2Vec_data.doc_vocab), self.embed_size).to(self.device)
        else:
            model = PV_DM(len(Doc2Vec_data.word_vocab),len(Doc2Vec_data.doc_vocab), self.embed_size,
                    self.window_size, self.hidden_size).to(self.device)
        #model = nn.DataParallel(model)

        # Optimizer.
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        return dataloader, model, optimizer, criterion, Doc2Vec_data

    def train(self,num_epochs=100,patience=15):
        losses = []
        lowest = 1e10
        wait = 0
        # Total number of training steps is number of batches * number of epochs.
        total_steps = self.dataset._len * num_epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        self.model.train()
        print('start training')
        for _e in tqdm(range(num_epochs)):
            epoch_loss=[]
            for batch in tqdm(self.dataloader):
                self.optimizer.zero_grad()
                doc = batch['doc'].to(self.device)
                context = batch['context'].to(self.device)
                if self.isNegSample:
                    target = batch['target'].to(self.device)
                    scores = self.model(doc, context,target)
                    loss = self.criterion(scores)
                else:
                    target = batch['target'].view(-1).to(self.device)
                    logprobs = self.model(doc,context)
                    loss = self.criterion(logprobs, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) #prevent the "exploding gradients" problem
                self.optimizer.step()
                epoch_loss.append(loss.item())
                scheduler.step() #Update the learning rate
            avg_loss = sum(epoch_loss) / len(epoch_loss)
            print(f'Epoch: {_e}, Loss: {avg_loss}')
            losses.append(avg_loss)
            if avg_loss < lowest:
                wait = 0
                lowest = avg_loss
                torch.save(self.model.state_dict(), './model/pvdm_checkpoint_{}.pt'.format(_e))
            else:
                wait += 1
                if wait > patience:
                    print(f'EarlyStopping at {_e} epoch out of {patience}')
                    break
        #plt.plot(losses)
        #plt.show()

    def evaluate(self,test_file,isvisualize=True):
        if self.isNegSample:
            print('Sorry, Negative Sample Model does not support evaluation')
            return
        true_positive = 0
        all_data = 0
        process_corpus = Process_corpus(window_size=self.window_size,isNegSample=self.isNegSample)
        tokenized_text_test,_,_ = process_corpus(test_file,None)
        self.model.eval()
        # Iterate through the test sentences.
        for sents in tqdm(tokenized_text_test):
            doc = sents[0]
            sent = sents[1]
            target = sents[2]
            vectorized_doc = self.dataset.doc_vectorize(doc).to(self.device)
            vectorized_sent = self.dataset.word_vectorize(sent).to(self.device)
            vectorized_target = self.dataset.word_vectorize(target).to(self.device)
            # Extract all the PVDM contexts (X) and targets (Y)
            # Retrieve the inputs and outputs.
            if -1 in vectorized_sent:  # Skip unknown words.
                continue

            with torch.no_grad():
                _, prediction = torch.max(self.model(vectorized_doc.unsqueeze(0),vectorized_sent.unsqueeze(0)), 1)
            true_positive += int(prediction) == int(vectorized_target)
            if isvisualize:
                self.visualize_predictions(vectorized_doc, vectorized_sent,vectorized_target, prediction)
            all_data += 1
        acc = true_positive/all_data
        print(f'Accuracy :{acc}')
        return acc

    def visualize_predictions(self,doc, context,target, prediction, unk='<unk>'):
        doc = self.dataset.doc_vocab.get(int(doc))
        left = ' '.join([self.dataset.word_vocab.get(int(_x), unk) for _x in context])
        target = self.dataset.word_vocab.get(int(target), unk)

        if not prediction:
            predicted_word = '______'
        else:
            predicted_word = self.dataset.word_vocab.get(int(prediction), unk)
        print(color_str(target, 'green'), '\t' if len(target) > 6 else '\t\t',
              doc, left, color_str(predicted_word, 'green' if target == predicted_word else 'red'))

if __name__ == '__main__':
    args = parser.parse_args()
    trainer = Trainer(file=args.input_dir,
                      batch_size=args.batch_size,
                      embed_size=args.embed_size,
                      window_size=args.window_size,
                      hidden_size=args.hidden_size,
                      learning_rate=args.learning_rate,
                      weight_decay=args.weight_decay,
                      isNegSample=args.isNegSample,
                      sample_size=args.sample_size)
    trainer.train(num_epochs=args.num_epochs,patience=args.patience)
    trainer.evaluate(test_file=args.test_dir)
