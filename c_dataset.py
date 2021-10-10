# Standard
import os
import sys

# PIP
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl

from transformers import BartTokenizer
from sentence_transformers import SentenceTransformer

# Custom


class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        batch_size=1,
        num_workers=0,
        option='all',
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.option = option
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.prepare_data()

    @staticmethod
    def encode_sentences(model_type,tokenizer, source_sentences, target_sentences, pad_to_max_length=True, return_tensors="pt"):
        ''' Function that tokenizes a sentence 
            Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
            Returns: Dictionary with keys: input_ids, attention_mask, target_ids
        '''

        input_ids = []
        attention_masks = []
        target_ids = []
        tokenized_sentences = {}

        if model_type == 'sbart':
            for sentence in source_sentences:
                encoded_dict = tokenizer(
                    sentence,
                    padding="max_length" if pad_to_max_length else 'longest',
                    truncation=True,
                    return_tensors=return_tensors,
                    add_prefix_space = True
                )

                input_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])

            input_ids = torch.cat(input_ids, dim = 0)
            attention_masks = torch.cat(attention_masks, dim = 0)

        else:
            model = SentenceTransformer('all-mpnet-base-v2')
            embeddings = model.encode(source_sentences)

        for sentence in target_sentences:
            encoded_dict = tokenizer(
                sentence,
                padding="max_length" if pad_to_max_length else 'longest',
                truncation=True,
                return_tensors=return_tensors,
                add_prefix_space = True
            )

            target_ids.append(encoded_dict['input_ids'])

        target_ids = torch.cat(target_ids, dim = 0)
  
        if model_type == 'sbart':
            batch = (input_ids,attention_masks,target_ids)
        else:
            batch = (torch.from_numpy(embeddings),target_ids)

        return batch

    def prepare_data(self):
        self.data = []

        dataset_dir = self.cfg.DATASET_DIR
        for dir in os.listdir(dataset_dir):
            dir_path = os.path.join(dataset_dir,dir)
            
            if not os.path.isdir(dir_path):
                continue

            files = [os.path.join(dir_path,f) for f in os.listdir(dir_path) if f.endswith('txt')]

            for f in files:
                data = open(f,'rb').read().decode(errors='replace').split('\n')
                self.data.extend(data)

        self.data = self.data[:1000]

        self.train, self.val, self.test = np.split(self.data,[int(.8*len(self.data)), int(.9*len(self.data))])

        self.train_dataset = self.encode_sentences(self.cfg.model_type,self.tokenizer,self.train,self.train)
        self.valid_dataset = self.encode_sentences(self.cfg.model_type,self.tokenizer,self.val,self.val)
        self.test_dataset = self.encode_sentences(self.cfg.model_type,self.tokenizer,self.test,self.test)

    def train_dataloader(self):
        dataset = TensorDataset(*self.train_dataset)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        dataset = TensorDataset(*self.valid_dataset)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        dataset = TensorDataset(*self.test_dataset)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

