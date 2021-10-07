# Standard
import os
import sys

# PIP
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import BartTokenizer


# Custom


class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        tokenizer,
        data_file,
        batch_size=1,
        num_workers=0,
        option='all',
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.option = option
        self.prepare_data()

    @staticmethod
    def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=32, pad_to_max_length=True, return_tensors="pt"):
    ''' Function that tokenizes a sentence 
        Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
        Returns: Dictionary with keys: input_ids, attention_mask, target_ids
    '''

    input_ids = []
    attention_masks = []
    target_ids = []
    tokenized_sentences = {}

    for sentence in source_sentences:
        encoded_dict = tokenizer(
            sentence,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            add_prefix_space = True
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)

    for sentence in target_sentences:
        encoded_dict = tokenizer(
            sentence,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            add_prefix_space = True
        )

        target_ids.append(encoded_dict['input_ids'])

    target_ids = torch.cat(target_ids, dim = 0)
  

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": target_ids,
    }

    return batch

    def prepare_data(self):
        self.data = open(self.data_file,'r').read().split('\n')
        self.train, self.val, self.test = np.split(self.data,[int(.8*len(self.data)), int(.9*len(self.data))])
        self.train_dataset = self.encode_sentences(self.tokenizer,self.train,self.train)
        self.valid_dataset = self.encode_sentences(self.tokenizer,self.val,self.val)
        self.test_dataset = self.encode_sentences(self.tokenizer,self.test,self.test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

