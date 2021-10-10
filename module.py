# Standard

# PIP
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import BartTokenizer
import numpy as np

# Custom
from c_transformers.models.bart.modeling_sbart import BartForConditionalGeneration,BartDecoder,BartConfig


class CustomModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-5,
        criterion_name='RMSE',
        optimizer_name='Adam',
        momentum=0.9,
        model_type='sbart',
        **kwargs
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        config = BartConfig.from_pretrained("facebook/bart-base")
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.decoder = BartDecoder(config,self.shared)
        
        self.model_type = model_type


        self.optimizer = self.get_optimizer(optimizer_name)

    def get_optimizer(self, optimizer_name):
        name = optimizer_name.lower()

        if name == 'SGD'.lower():
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        if name == 'Adam'.lower():
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if name == 'AdamW'.lower():
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        raise ValueError(f'{optimizer_name} is not on the custom optimizer list!')

    def forward(self, **kwargs):
        
        '''
        if self.model_type == 'sbart':
            out = self.model(**kwargs)
        else:
            out = self.decoder(**kwargs)
        '''

        out = self.model(**kwargs)

        return out

    def step(self, batch, batch_idx):
        if self.model_type == 'sbart':
            data, mask, target = batch

            output = self(
                input_ids=data, 
                attention_mask = mask, 
                decoder_input_ids = target
            )

        else:
            data, target = batch
            '''
            output_attentions = self.config.output_attentions
            output_hidden_states = self.config.output_hidden_states
            use_cache = self.config.use_cache
            return_dict = self.config.use_return_dict
            output = self(
                input_ids=target, 
                encoder_hidden_states = data,
                use_cache = use_cache,
                output_attentions = output_attentions
                output_hidden_states = output_hidden_states
                return_dict = return_dict,
            )
            '''
            output = self(
                inputs_embeds = data,
                decoder_input_ids = target
            )

        logits = output[0]
        ce_loss_func = nn.CrossEntropyLoss(ignore_index = self.tokenizer.pad_token_id)

        loss = ce_loss_func(logits.view(-1,logits.shape[-1]),target.view(-1))

        return {
            'loss': loss,
        }

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def epoch_end(self, outputs, state='train'):
        loss = torch.tensor(0, dtype=torch.float)
        loss_arr = []

        for i in outputs:
            loss_cpu = i['loss'].cpu().detach()
            loss += loss_cpu
            loss_arr.append(float(loss_cpu))
        loss = loss / len(outputs)

        self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
        
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='val')

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, state='test')
