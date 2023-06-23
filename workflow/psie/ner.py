#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset
from transformers import BertForTokenClassification

from psie.utils import toBertNer

from psie.data import preprocess_text

class BertForNer(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)

    def finetuning(self, train_loader, val_loader, device, max_norm, optimizer):
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []

        self.train()

        for idx, batch in enumerate(train_loader):

            ids = batch["input_ids"].to(device, dtype=torch.long)
            mask = batch["attention_mask"].to(device, dtype=torch.long)
            labels = batch["labels"].to(device, dtype=torch.long)

            outputs = self(input_ids=ids, attention_mask=mask, labels=labels)

            loss = outputs[0]
            tr_logits = outputs[1]
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)

            flattened_targets = labels.view(-1) 
            active_logits = tr_logits.view(
                -1, self.num_labels
            )  
            flattened_predictions = torch.argmax(
                active_logits, axis=1
            )  

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 
            
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tr_labels.extend(labels)
            tr_preds.extend(predictions)

            tmp_tr_accuracy = accuracy_score(
                labels.cpu().numpy(), predictions.cpu().numpy()
            )
            tr_accuracy += tmp_tr_accuracy

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=self.parameters(), max_norm=max_norm
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")

        self.eval()

        val_loss = 0
        for val_idx, val_batch in enumerate(val_loader):
            val_ids = val_batch["input_ids"].to(device, dtype=torch.long)
            val_mask = val_batch["attention_mask"].to(device, dtype=torch.long)
            val_labels = val_batch["labels"].to(device, dtype=torch.long)

            outputs = self(
                input_ids=val_ids, attention_mask=val_mask, labels=val_labels
            )
            loss = outputs[0]
            val_loss += loss.item()
        print(f"Val loss s: {val_loss/val_idx}")

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

    def testLabeledData(self, test_loader, device, id_to_BOI):
        self.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        eval_preds, eval_labels = [], []

        with torch.no_grad():
            for idx, batch in enumerate(test_loader):

                ids = batch["input_ids"].to(device, dtype=torch.long)
                mask = batch["attention_mask"].to(device, dtype=torch.long)
                labels = batch["labels"].to(device, dtype=torch.long)

                outputs = self(input_ids=ids, attention_mask=mask, labels=labels)
                loss = outputs[0]
                eval_logits = outputs[1]

                eval_loss += loss.item()

                nb_eval_steps += 1
                nb_eval_examples += labels.size(0)

                if idx % 100 == 0:
                    loss_step = eval_loss / nb_eval_steps
                    print(f"Validation loss per 100 evaluation steps: {loss_step}")

                flattened_targets = labels.view(-1)  
                active_logits = eval_logits.view(
                    -1, self.num_labels
                )  
                flattened_predictions = torch.argmax(
                    active_logits, axis=1
                )  

                # only compute accuracy at active labels
                active_accuracy = labels.view(-1) != -100  

                labels = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(
                    flattened_predictions, active_accuracy
                )

                eval_labels.extend(labels)
                eval_preds.extend(predictions)

                tmp_eval_accuracy = accuracy_score(
                    labels.cpu().numpy(), predictions.cpu().numpy()
                )
                eval_accuracy += tmp_eval_accuracy

        labels = [id_to_BOI[id.item()] for id in eval_labels]
        predictions = [id_to_BOI[id.item()] for id in eval_preds]

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        print(f"Validation Loss: {eval_loss}")
        print(f"Validation Accuracy: {eval_accuracy}")

        return labels, predictions

    def predict(self, dataloader, device, id_to_BOI):
        self.eval()

        pred_array = []
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                ids = batch["input_ids"].to(device, dtype=torch.long)
                mask = batch["attention_mask"].to(device, dtype=torch.long)

                outputs = self(input_ids=ids, attention_mask=mask)
                eval_logits = outputs[0]

                pred_array.append(
                    torch.argmax(eval_logits, axis=2)
                )  # shape (batch_size * seq_len,)

        for i in range(len(pred_array)):
            for pred in pred_array[i].cpu().numpy():
                predictions.append([id_to_BOI[id.item()] for id in pred])

        return predictions


class NewNerLabeledDataset(Dataset):
    def __init__(self, data, tokenizer, entities, BOI_to_id, max_len=256):

        self.max_len = max_len
        self.BOI_to_id = BOI_to_id

        IOBs = toBertNer(data, tokenizer, entities, padding=True, max_len=self.max_len)
        self.len = len(IOBs["sentence"])

        self.sentences = IOBs["sentence"]
        self.labels = IOBs["labels"]
        self.tokenizer = tokenizer

    def __getitem__(self, index):

        sentence = self.sentences[index].strip()
        encoded_labels = [self.BOI_to_id[iob] for iob in self.labels[index]]

        encoding = self.tokenizer(
            sentence, padding="max_length", max_length=self.max_len
        )

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(encoded_labels)

        return item

    def __len__(self):
        return self.len

class NerLabeledDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, BOI_to_id):

        self.max_len = max_len
        self.BOI_to_id = BOI_to_id

        IOBs = data.get_token_entities(tokenizer, padding=True, max_len=self.max_len)
        self.len = len(IOBs["sentences"])

        self.sentences = IOBs["sentences"]
        self.labels = IOBs["labels"]
        self.tokenizer = tokenizer

    def __getitem__(self, index):

        sentence = self.sentences[index].strip()
        encoded_labels = [self.BOI_to_id[iob] for iob in self.labels[index]]

        encoding = self.tokenizer(
            sentence, padding="max_length", max_length=self.max_len
        )

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(encoded_labels)

        return item

    def __len__(self):
        return self.len


class NerUnlabeledDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):

        self.max_len = max_len
        self.len = len(data)
        self.sentences = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):

        sentence = self.sentences[index].strip()

        encoding = self.tokenizer(
            preprocess_text(sentence),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["plain"] = (
            sentence.replace("$", "")
            .replace("_", "")
            .replace("}", "")
            .replace("{", "")
            .replace("~", "")
            .replace("\\", "")
        )

        return item

    def __len__(self):
        return self.len
