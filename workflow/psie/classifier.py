#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertModel
from bs4 import BeautifulSoup
import re

from psie.data import get_entities


from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2

        self.bert = BertModel.from_pretrained('m3rg-iitd/matscibert')

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        
        return logits

    def finetuning(self, train_loader, val_loader, device, max_norm, optimizer, weight=[1, 1]):
        # 1 epoch finetuning
        
        total_loss_tr = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        criterion = nn.CrossEntropyLoss(weight=weight).to(device)
        
        self.train()
        
        for idx, batch in enumerate(train_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['isrelevant'].to(device, dtype = torch.long).view(-1)

            predictions = self(input_ids=ids, attention_mask=mask).view(-1, 2)

            batch_loss = criterion(predictions, labels)
            total_loss_tr += batch_loss.item()

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=self.parameters(), max_norm=max_norm
            )
            
            # backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if idx % 100==0:
                loss_step = total_loss_tr/nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")
                
        epoch_loss = total_loss_tr / nb_tr_steps        
        print(f"Training loss epoch: {epoch_loss}")
        
    def testLabeledData(self, test_loader, device):
        self.eval()
        eval_preds, eval_labels = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                
                ids = batch['input_ids'].to(device, dtype = torch.long)
                mask = batch['attention_mask'].to(device, dtype = torch.long)                
                labels = batch['isrelevant'].to(device, dtype = torch.long).view(-1)
                
                predictions = self(input_ids=ids, attention_mask=mask).view(-1, 2)
                
                eval_labels.extend(labels)
                eval_preds.extend(predictions)
                
        return eval_labels, eval_preds

    def predict(self, dataloader, device):
        self.eval()
        eval_preds = []
        
        with torch.no_grad():
            for batch in dataloader:
                
                ids = batch['input_ids'].to(device, dtype = torch.long)
                mask = batch['attention_mask'].to(device, dtype = torch.long)
                
                predictions = self(input_ids=ids, attention_mask=mask).view(-1, 2)           
                eval_preds.extend(predictions)
                
        return eval_preds