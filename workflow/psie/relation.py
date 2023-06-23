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

class BertForRelations(nn.Module):
    def __init__(self, pretrained, dropout=0.5, use_cls_embedding=True):
        super(BertForRelations, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained)
        self.dropout = nn.Dropout(dropout)

        if use_cls_embedding:
            self.linear = nn.Linear(768, 2)
        else:
            self.linear = nn.Linear(2*768, 2)

        self.activation = nn.Sigmoid()
        self.use_cls_embedding = use_cls_embedding

    def forward(self, input_ids, attention_mask, tags):
        outputs = self.bert(input_ids= input_ids, attention_mask=attention_mask, return_dict=False)

        if self.use_cls_embedding:
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.linear(pooled_output)
        else:
            e1_output = outputs[0][range(tags.size(0)), tags[:,0]]
            e2_output = outputs[0][range(tags.size(0)), tags[:,1]]
            ent_output = self.dropout(torch.cat((e1_output, e2_output), dim=1))
            logits = self.linear(ent_output)        

        return logits

    def finetuning(self, train_loader, val_loader, device, max_norm, optimizer, weight=None):
        total_loss_tr = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        if weight is None:
            criterion = nn.CrossEntropyLoss().to(device)
        else:
            criterion = nn.CrossEntropyLoss(weight=weight).to(device)
        
        self.train()
        
        for idx, batch in enumerate(train_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            tags = batch['tag_positions'].to(device, dtype = torch.long)
            labels = batch['isrelated'].to(device, dtype = torch.long).view(-1)

            predictions = self(input_ids=ids, attention_mask=mask, tags=tags).view(-1, 2)

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
        
        self.eval()

        val_loss = 0
        with torch.no_grad():
          for val_idx, val_batch in enumerate(val_loader):
              val_ids = val_batch['input_ids'].to(device, dtype = torch.long)
              val_mask = val_batch['attention_mask'].to(device, dtype = torch.long)
              val_tags = val_batch['tag_positions'].to(device, dtype = torch.long)
              val_labels = val_batch['isrelated'].to(device, dtype = torch.long).view(-1)

              val_pred = self(input_ids=val_ids, attention_mask=val_mask, tags=val_tags).view(-1, 2)
              batch_val_loss = criterion(val_pred, val_labels)      
              val_loss += batch_val_loss.item()
        print(f"Val loss s: {val_loss/val_idx}")
        
        epoch_loss = total_loss_tr / nb_tr_steps        
        print(f"Training loss epoch: {epoch_loss}")
        
    def testLabeledData(self, test_loader, device):
        self.eval()
        eval_preds, eval_labels = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                
                ids = batch['input_ids'].to(device, dtype = torch.long)
                mask = batch['attention_mask'].to(device, dtype = torch.long)
                tags = batch['tag_positions'].to(device, dtype = torch.long)
                labels = batch['isrelated'].to(device, dtype = torch.long).view(-1)
                
                predictions = self(input_ids=ids, attention_mask=mask, tags=tags).view(-1, 2)
                
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
                tags = batch['tag_positions'].to(device, dtype = torch.long)
                
                predictions = self(input_ids=ids, attention_mask=mask, tags=tags).view(-1, 2)           
                eval_preds.extend(predictions)
                
        return eval_preds

class RelationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):

        self.max_len = max_len

        self.len = len(data["sentence"])

        self.sentences = data["sentence"]
        self.isrelated = data["isrelated"]
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        sentence = self.sentences[index].strip()

        encoding = self.tokenizer(
            sentence, padding="max_length", max_length=self.max_len, truncation=True
        )

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["tag_positions"] = torch.as_tensor([self.tokenizer.tokenize(sentence).index("[e1]")+1, self.tokenizer.tokenize(sentence).index("[e2]")+1])

        if self.isrelated[index] is not None:
            item["isrelated"] = torch.as_tensor(self.isrelated[index])
        else:
            item["isrelated"] = torch.as_tensor(0)

        return item

    def __len__(self):
        return self.len

    @classmethod
    def fromNlpData(cls, dataset, tokenizer, max_len):

        data = {"sentence": [], "isrelated": []}

        for key in sorted(dataset.keys()):
            sentence = BeautifulSoup(dataset[key].text, "html.parser").find_all("body")[0].get_text()
            
            sentence = (sentence.replace("$", "")
                .replace("_", "")
                .replace("}", "")
                .replace("{", "")
                .replace("~", "")
                .replace("\\", "")
            )

            entities = psie.get_entities(dataset[key].text)
            relations = [dataset[key].compound, dataset[key].Tc]
            
            entry_rel = cls._generateRelationsData(sentence, entities, relations)

            data["sentence"].extend(entry_rel["sentence"])
            data["isrelated"].extend(entry_rel["isrelated"])

        return cls(data, tokenizer, max_len)

    @classmethod
    def fromBertNer(cls, sentences, entities):
        pass

    @staticmethod
    def _generateRelationsData(sentence, entities, relations=None):

        rel_dict = {"sentence": [], "isrelated": [], "indx": []}
        
        for comp in entities["chemical-entity"]:

            comp = (comp.replace("$", "")
            .replace("_", "")
            .replace("}", "")
            .replace("{", "")
            .replace("~", "")
            .replace("\\", "")
            )
           
            sentence_1 = re.sub(
                            "[^a-zA-Z0-9]"+re.escape(comp)+"+[^a-zA-Z0-9]",
                            " [E1]"+comp+"[/E1] ",
                            sentence, 1
                        )

            if relations is not None:
                comp_list = [c.strip() for c in reversed(relations[0])]
                if comp in comp_list:
                    i = comp_list.index(comp)
                else:
                    i = None

            for temp in entities["temperature"]:

                temp = (temp.replace("$", "")
                .replace("_", "")
                .replace("}", "")
                .replace("{", "")
                .replace("~", "")
                .replace("\\", "")
                )

                indx_2 = sentence_1.index(temp)
                sentence_2 = (
                    sentence_1[0:indx_2]
                    + "[E2]"
                    + sentence_1[indx_2 : indx_2 + len(temp)]
                    + "[/E2]"
                    + sentence_1[indx_2 + len(temp):]
                )

                if relations is not None:
                    temp_list = [t.strip() for t in reversed(relations[1])]
                    if temp in temp_list:
                        j = temp_list.index(temp)
                    else:
                        j = None

                    if (i == j) and (i is not None) and (j is not None):
                        isrelated = 1
                    else:
                        isrelated = 0
                else:
                    isrelated = None

                rel_dict["sentence"].append(sentence_2)                
                rel_dict["isrelated"].append(isrelated)

                if relations is not None:
                    rel_dict["indx"].append([i, j])
                else:
                    rel_dict["indx"].append(None)

        return rel_dict

