

!pip install -U torch==1.8.0+cu111 torchtext==0.9.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html



!pip install datasets

import datasets as ds
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import json
from collections import defaultdict
import re
import numpy as np
from torchtext.vocab import Vectors
import nltk
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import nltk
nltk.download('punkt')

nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
# Commented out IPython magic to ensure Python compatibility.
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# %matplotlib inline
from torchtext.data.utils import get_tokenizer
from torchtext.legacy import data
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
data_nli=ds.load_dataset("multi_nli")
TEXT=data.Field(use_vocab=True, lower=True, batch_first=True, include_lengths=True)
LABEL = data.LabelField(dtype=torch.long, batch_first=True, sequential=False)

def _clean(i):
        i=i.lower()
        i = re.sub("[`]","", i)
        i = re.sub("''+","", i)
        i = re.sub("[^A-Za-z0-9' ]","", i)
        return i
def clean(data):
  premise=[]
  hypothesis=[]
  label=[]
  STOPWORDS = stopwords.words('english')
  for i in data:
    words=[token for token in tokenizer(_clean(i['premise'])) if len(token)>=1 and token not in STOPWORDS and token != " "]
    hyp=[token for token in tokenizer(_clean(i['hypothesis'])) if len(token)>=1 and token not in STOPWORDS and token != " "]
    words=["<sos>"]+words
    words.append("<eos>")
    hyp=["<sos>"]+hyp
    hyp.append("<eos>")
    if len(words)>0:
      premise.append(words)
      hypothesis.append(hyp)
      label.append(torch.tensor(i['label']).view(1))
    else:
      print(i)
  return premise,hypothesis,label
def get_data(dataset, vocab):
    data = []
    for sentence in dataset:
            tokens = [vocab[token] for token in sentence]
            data.append(torch.LongTensor(tokens))
    return data
def build(data):
    for i in data:
      yield i
data_train=data_nli['train']
data_validation_matched=data_nli['validation_matched']
data_validation_mismatched=data_nli['validation_mismatched']
train_premise,train_hypothesis,train_labels=clean(list(data_train)[:10000])
val_match_premise,val_match_hypothesis,val_match_labels=clean(list(data_validation_matched)[:2000])
val_mismatch_premise,val_mismatch_hypothesis,val_mismatch_labels=clean(list(data_validation_mismatched)[:2000])
TEXT.build_vocab(train_premise,
                 vectors = "glove.6B.300d", 
                 unk_init = torch.Tensor.normal_)
train_premise_nli=get_data(train_premise,TEXT.vocab)
train_hypothesis_nli=get_data(train_hypothesis,TEXT.vocab)
val_match_premise_nli=get_data(val_match_premise,TEXT.vocab)
val_match_hypothesis_nli=get_data(val_match_hypothesis,TEXT.vocab)
val_mismatch_premise_nli=get_data(val_mismatch_premise,TEXT.vocab)
val_mismatch_hypothesis_nli=get_data(val_mismatch_hypothesis,TEXT.vocab)

len(data_train[0])

class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(ELMo, self).__init__()
        # glove_vectors= GloVe()
        self.embedding_dim=embedding_dim
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                            batch_first=True,bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=2*hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                             batch_first=True,bidirectional=True)
        self.linear = nn.Linear(2*hidden_dim,vocab_size)
        self.gamma = nn.Parameter(torch.ones(1))
        self.s = nn.Parameter(torch.zeros(3))
    def forward(self,embeds):
        
        lstm_1, _ = self.lstm(embeds)
        lstm_2, _ = self.lstm2(lstm_1)
        # Combine the ELMo representations from each layer
        lstm_embed_1=lstm_1.view(lstm_1.shape[0],lstm_1.shape[1],1,lstm_1.shape[2])
        lstm_embed_2=lstm_2.view(lstm_2.shape[0],lstm_2.shape[1],1,lstm_2.shape[2])
        embeds=embeds.view(embeds.shape[0],embeds.shape[1],1,embeds.shape[2])
        elmo_representations = torch.cat([lstm_embed_1, lstm_embed_2,embeds], dim=2)
        elmo_weights = F.softmax(self.s, dim=0).repeat(len(embeds),1,1,1).view(len(embeds),1,-1,1)  # (1, num_layers*2, 1)
        weighted_sum = torch.sum( elmo_representations*elmo_weights, dim=2)  # (seq_len, batch_size, 2*hidden_dim)
        weighted_sum = self.gamma * weighted_sum
        out = F.softmax(self.linear(weighted_sum))
        return out,weighted_sum

class NLI(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(NLI, self).__init__()
        # glove_vectors= GloVe()
        self.embedding_dim=embedding_dim
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                            batch_first=True,dropout=0.5,bidirectional=True)
        self.hidden_2_hidden=nn.Linear(4*hidden_dim,hidden_dim)
        self.hidden_2_sof=nn.Linear(hidden_dim,3)
        self.sof=nn.Softmax()
    def forward(self,embeds_premise,embeds_hypo):
        lstm_p, _ = self.lstm(embeds_premise)
        lstm_h, _ = self.lstm(embeds_hypo)
        lstm_p = lstm_p[:,0,:]
        lstm_h = lstm_h[:,0,:]
        
        avg_f= torch.mean(torch.stack([lstm_p,lstm_h],dim=1),1)
        out = self.hidden_2_hidden(torch.cat([lstm_p-avg_f,lstm_h-avg_f],dim=1))
        out = F.relu(out)
        out = self.hidden_2_sof(out)
        out = self.sof(out)
        return out

EMBEDDING_DIM = 150
HIDDEN_DIM = 150
EPOCHS = 30
LEARNING_RATE=5e-4
NUMBER_OF_LAYERS=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=ELMo(len(TEXT.vocab),2*EMBEDDING_DIM,HIDDEN_DIM)
model=model.to(device)
train_batch_size=128
test_batch_szie=16
loss_function = nn.CrossEntropyLoss()
loss_function=loss_function.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
dataloader_train = DataLoader(list(zip(pad_sequence(train_premise_nli, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),pad_sequence(train_hypothesis_nli, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),train_labels)), batch_size=train_batch_size)
dataloader_val_match = DataLoader(list(zip(pad_sequence(val_match_premise_nli, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),pad_sequence(val_match_hypothesis_nli, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),val_match_labels)), batch_size=test_batch_szie)
dataloader_val_mismatch = DataLoader(list(zip(pad_sequence(val_mismatch_premise_nli, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),pad_sequence(val_mismatch_hypothesis_nli, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),val_mismatch_labels)), batch_size=test_batch_szie)
def train():
  for epoch in range(EPOCHS):
          train_loss=0
          tags=[]
          y_true=[]
          best_valid_loss = float('inf')
          for batch in tqdm(dataloader_train):
            model.zero_grad()
            vectors_p,vectors_h=TEXT.vocab.vectors[batch[0][:,:-1]],TEXT.vocab.vectors[batch[1][:,:-1]]
            vectors_p,vectors_h=vectors_p.to(device),vectors_h.to(device)
            y_actual_p,y_actual_h=batch[0][:,1:],batch[1][:,1:]
            y_actual_p,y_actual_h=y_actual_p.to(device),y_actual_h.to(device)
            y_pred_p,embeds_p = model(vectors_p)
            y_pred_h,embeds_h = model(vectors_h)
            y_pred_p,y_pred_h=y_pred_p.reshape(y_actual_p.shape[0]*y_actual_p.shape[1],-1),y_pred_h.reshape(y_actual_h.shape[0]*y_actual_h.shape[1],-1)
            y_actual_p,y_actual_h=y_actual_p.reshape(-1),y_actual_h.reshape(-1)
            loss = loss_function(y_pred_p,y_actual_p)+loss_function(y_pred_h,y_actual_h)
            train_loss+=loss.item()
            loss.backward()
            optimizer.step()
          print("train_loss",epoch,train_loss)
          train_loss=0
          with torch.no_grad():
            for batch in tqdm(dataloader_val_match):
              
              vectors_p,vectors_h=TEXT.vocab.vectors[batch[0][:,:-1]],TEXT.vocab.vectors[batch[1][:,:-1]]
              vectors_p,vectors_h=vectors_p.to(device),vectors_h.to(device)
              y_actual_p,y_actual_h=batch[0][:,1:],batch[1][:,1:]
              y_actual_p,y_actual_h=y_actual_p.to(device),y_actual_h.to(device)
              y_pred_p,embeds_p = model(vectors_p)
              y_pred_h,embeds_h = model(vectors_h)
              y_pred_p,y_pred_h=y_pred_p.reshape(y_actual_p.shape[0]*y_actual_p.shape[1],-1),y_pred_h.reshape(y_actual_h.shape[0]*y_actual_h.shape[1],-1)
              y_actual_p,y_actual_h=y_actual_p.reshape(-1),y_actual_h.reshape(-1)
              loss = loss_function(y_pred_p,y_actual_p)+loss_function(y_pred_h,y_actual_h)
              train_loss+=loss.item()
            print("val_loss",epoch,train_loss)
            train_loss=0
            for batch in tqdm(dataloader_val_mismatch):
              
              vectors_p,vectors_h=TEXT.vocab.vectors[batch[0][:,:-1]],TEXT.vocab.vectors[batch[1][:,:-1]]
              vectors_p,vectors_h=vectors_p.to(device),vectors_h.to(device)
              y_actual_p,y_actual_h=batch[0][:,1:],batch[1][:,1:]
              y_actual_p,y_actual_h=y_actual_p.to(device),y_actual_h.to(device)
              y_pred_p,embeds_p = model(vectors_p)
              y_pred_h,embeds_h = model(vectors_h)
              y_pred_p,y_pred_h=y_pred_p.reshape(y_actual_p.shape[0]*y_actual_p.shape[1],-1),y_pred_h.reshape(y_actual_h.shape[0]*y_actual_h.shape[1],-1)
              y_actual_p,y_actual_h=y_actual_p.reshape(-1),y_actual_h.reshape(-1)
              loss = loss_function(y_pred_p,y_actual_p)+loss_function(y_pred_h,y_actual_h)
              train_loss+=loss.item()
              
            print("val_mis_loss",epoch,train_loss)
train()
torch.save(model.state_dict(), 'elmo_nli.pt')

from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from torch.nn.utils.rnn import pad_sequence
EMBEDDING_DIM = 150
HIDDEN_DIM = 150
EPOCHS = 10
LEARNING_RATE=5e-4
NUMBER_OF_LAYERS=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_batch_size=64
test_batch_szie=16
loss_function = nn.CrossEntropyLoss()
loss_function=loss_function.to(device)

dataloader_train = DataLoader(list(zip(pad_sequence(train_premise_nli, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),pad_sequence(train_hypothesis_nli, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),train_labels)), batch_size=train_batch_size)
dataloader_val_match = DataLoader(list(zip(pad_sequence(val_match_premise_nli, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),pad_sequence(val_match_hypothesis_nli, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),val_match_labels)), batch_size=test_batch_szie)
dataloader_val_mismatch = DataLoader(list(zip(pad_sequence(val_mismatch_premise_nli, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),pad_sequence(val_mismatch_hypothesis_nli, batch_first=True,padding_value=TEXT.vocab.stoi['<pad>']),val_mismatch_labels)), batch_size=test_batch_szie)
model_nli=NLI(len(TEXT.vocab),2*EMBEDDING_DIM,HIDDEN_DIM)
model_nli=model_nli.to(device)
optimizer = torch.optim.Adam(model_nli.parameters(), lr=LEARNING_RATE)
def train():
  _train_loss=[]
  _val_match_loss=[]
  _val_mismatch_loss=[]
  for epoch in range(EPOCHS):
          train_loss=0
          y_true=[]
          y_preds=[]
          for batch in tqdm(dataloader_train):
            model.zero_grad()
            vectors_p,vectors_h=TEXT.vocab.vectors[batch[0]],TEXT.vocab.vectors[batch[1]]
            vectors_p,vectors_h=vectors_p.to(device),vectors_h.to(device)
            y_actual=batch[2].view(-1)
            y_actual=y_actual.to(device)
            y_pred_p,elmo_embeds_p = model(vectors_p)
            y_pred_h,elmo_embeds_h = model(vectors_h)
            y_pred=model_nli(elmo_embeds_p,elmo_embeds_h)
            #F.one_hot(y_actual,num_classes=3)
            indices = torch.max(y_pred, 1)[1]
            y_preds+=list(np.rint(indices.view(-1).detach().cpu().numpy()))
            y_true+=list(np.rint(y_actual.detach().cpu().numpy()))
            loss = loss_function(y_pred,y_actual)
            train_loss+=loss.item()
            loss.backward()
            optimizer.step()
          print("train_loss",epoch,train_loss)
          _train_loss.append(train_loss)
          print(classification_report(y_true,y_preds))
          
          with torch.no_grad():
              best_valid_loss = float('inf')
              y_true=[]
              y_preds=[] 
              valid_loss_match=0
              for batch in tqdm(dataloader_val_match):
                vectors_p,vectors_h=TEXT.vocab.vectors[batch[0]],TEXT.vocab.vectors[batch[1]]
                vectors_p,vectors_h=vectors_p.to(device),vectors_h.to(device)
                y_actual=batch[2].view(-1)
                y_actual=y_actual.to(device)
                y_pred_p,elmo_embeds_p = model(vectors_p)
                y_pred_h,elmo_embeds_h = model(vectors_h)
                y_pred=model_nli(elmo_embeds_p,elmo_embeds_h)
                indices = torch.max(y_pred, 1)[1]
                y_preds+=list(np.rint(indices.view(-1).detach().cpu().numpy()))
                y_true+=list(np.rint(y_actual.detach().cpu().numpy()))
                #F.one_hot(y_actual,num_classes=3)
                loss = loss_function(y_pred,y_actual)
                valid_loss_match+=loss.item()
              print("valid_loss_match",valid_loss_match)
              _val_match_loss.append(valid_loss_match)
              print(classification_report(y_true,y_preds))
              # ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true,y_preds)).plot()
              valid_loss_mismatch=0
              y_true=[]
              y_preds=[] 
              for batch in tqdm(dataloader_val_mismatch):
                vectors_p,vectors_h=TEXT.vocab.vectors[batch[0]],TEXT.vocab.vectors[batch[1]]
                vectors_p,vectors_h=vectors_p.to(device),vectors_h.to(device)
                y_actual=batch[2].view(-1)
                y_actual=y_actual.to(device)
                y_pred_p,elmo_embeds_p = model(vectors_p)
                y_pred_h,elmo_embeds_h = model(vectors_h)
                y_pred=model_nli(elmo_embeds_p,elmo_embeds_h)
                #F.one_hot(y_actual,num_classes=3)
                indices = torch.max(y_pred, 1)[1]
                y_preds+=list(np.rint(indices.view(-1).detach().cpu().numpy()))
                y_true+=list(np.rint(y_actual.detach().cpu().numpy()))
                loss = loss_function(y_pred,y_actual)
                valid_loss_mismatch+=loss.item()
              print("valid_loss_mismatch",valid_loss_mismatch)
              print(classification_report(y_true,y_preds))
              _val_mismatch_loss.append(valid_loss_mismatch)
              # ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true,y_preds)).plot()
              if  min(valid_loss_mismatch,valid_loss_match) < best_valid_loss:
                  best_valid_loss =  min(valid_loss_mismatch,valid_loss_match)
                  torch.save(model.state_dict(), 'nli.pt')
  plt.plot(np.arange(1, EPOCHS + 1), _train_loss, label='Train Loss', color='green')
  plt.plot(np.arange(1, EPOCHS + 1), _val_mismatch_loss, label='Validation Mismatch', color='red')
  plt.plot(np.arange(1, EPOCHS + 1), _val_match_loss, label='Validation Match', color='blue')
  plt.xlabel('EPOCH->')
  plt.ylabel('LOSS->')
  plt.legend()
  plt.show()
train()

import pandas as pd
premise=[]
hy=[]
for batch in tqdm(dataloader_train):
    vectors_p,vectors_h=TEXT.vocab.vectors[batch[0]],TEXT.vocab.vectors[batch[1]]
    vectors_p,vectors_h=vectors_p.to(device),vectors_h.to(device)
    y_actual=batch[2].view(-1)
    y_actual=y_actual.to(device)
    y_pred_p,elmo_embeds_p = model(vectors_p)
    y_pred_h,elmo_embeds_h = model(vectors_h)
    premise.append(elmo_embeds_p.detach().cpu().numpy())
    hy.append(elmo_embeds_h.detach().cpu().numpy())
pr=pd.DataFrame(np.array(premise))
h=pd.DataFrame(np.array(hy))

pr.to_csv("embedding_premise.csv")
h.to_csv("embedding_hypothesis.csv")

def test():
  model.load_state_dict(torch.load('nli.pt'))
  
  with torch.no_grad():
      best_valid_loss = float('inf')
      y_true=[]
      y_preds=[] 
      valid_loss_match=0
      for batch in tqdm(dataloader_val_match):
        vectors_p,vectors_h=TEXT.vocab.vectors[batch[0]],TEXT.vocab.vectors[batch[1]]
        vectors_p,vectors_h=vectors_p.to(device),vectors_h.to(device)
        y_actual=batch[2].view(-1)
        y_actual=y_actual.to(device)
        y_pred_p,elmo_embeds_p = model(vectors_p)
        y_pred_h,elmo_embeds_h = model(vectors_h)
        y_pred=model_nli(elmo_embeds_p,elmo_embeds_h)
        indices = torch.max(y_pred, 1)[1]
        y_preds+=list(np.rint(indices.view(-1).detach().cpu().numpy()))
        y_true+=list(np.rint(y_actual.detach().cpu().numpy()))
        #F.one_hot(y_actual,num_classes=3)
        loss = loss_function(y_pred,y_actual)
        valid_loss_match+=loss.item()
      print("valid_loss_match",valid_loss_match)
      print(classification_report(y_true,y_preds))
      ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true,y_preds)).plot()
      valid_loss_mismatch=0
      y_true=[]
      y_preds=[] 
      for batch in tqdm(dataloader_val_mismatch):
        vectors_p,vectors_h=TEXT.vocab.vectors[batch[0]],TEXT.vocab.vectors[batch[1]]
        vectors_p,vectors_h=vectors_p.to(device),vectors_h.to(device)
        y_actual=batch[2].view(-1)
        y_actual=y_actual.to(device)
        y_pred_p,elmo_embeds_p = model(vectors_p)
        y_pred_h,elmo_embeds_h = model(vectors_h)
        y_pred=model_nli(elmo_embeds_p,elmo_embeds_h)
        #F.one_hot(y_actual,num_classes=3)
        indices = torch.max(y_pred, 1)[1]
        y_preds+=list(np.rint(indices.view(-1).detach().cpu().numpy()))
        y_true+=list(np.rint(y_actual.detach().cpu().numpy()))
        loss = loss_function(y_pred,y_actual)
        valid_loss_mismatch+=loss.item()
      print("valid_loss_mismatch",valid_loss_mismatch)
      print(classification_report(y_true,y_preds))
      ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true,y_preds)).plot()
test()

