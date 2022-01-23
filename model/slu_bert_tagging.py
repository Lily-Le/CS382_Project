#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from transformers import BertTokenizer,BertModel
import time

class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        # self.cell = config.encoder_cell
        # self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        # self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        
        bert_path = "./BERT/chinese_roberta_wwm_ext_pytorch"
        self.bert = BertModel.from_pretrained(bert_path)

        #for param in self.bert.parameters():
        #    param.requires_grad = True

        self.lstm = nn.LSTM(768, config.hidden_size // 2, config.num_layer,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
      
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)


    def forward(self, batch):

    	# with BERT
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        input_masks = batch.input_masks

        outputs = self.bert(input_ids, attention_mask=input_masks )
        last_hidden_states = outputs.last_hidden_state
        lstm_out, _ = self.lstm(last_hidden_states)
        my_hiddens = self.dropout_layer(lstm_out)
        my_tag_output = self.output_layer(my_hiddens, tag_mask, tag_ids)   
        
        
        return my_tag_output


    def forward_test(self, batch):

    	# with BERT
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        input_masks = batch.input_masks

        outputs = self.bert(input_ids, attention_mask=input_masks )
        last_hidden_states = outputs.last_hidden_state
        lstm_out, _ = self.lstm(last_hidden_states)
        my_hiddens = self.dropout_layer(lstm_out)
        my_tag_output = self.output_layer(my_hiddens, tag_mask)   
        
        
        return my_tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        return predictions, labels, loss.cpu().item()

    def decode_test(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob = self.forward_test(batch)
        predictions = []
        test_result = batch.examples.copy()
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    act = tag_buff[0].split('-')[1]
                    slot = tag_buff[0].split('-')[2]
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append([act,slot,value])
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                act = tag_buff[0].split('-')[1]
                slot = tag_buff[0].split('-')[2]
                # slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                # pred_tuple.append(f'{slot}-{value}')
                pred_tuple.append([act,slot, value])
            predictions.append(pred_tuple)
        return predictions, labels



class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        # print('num_tags',num_tags)
        # print('--------output_layer input_size:',input_size)
        print('pad_id',pad_id)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        # print('--------output_layer logits:',logits.shape)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        
        #print('logits', logits.shape)
        #print('labels shape', labels.shape)
        #print('labels', labels[2])
        #m = logits.view(-1, logits.shape[-1])
        #n = labels.view(-1)
        #print('m',m.shape)
        #print('n',n.shape)
        prob = torch.softmax(logits, dim=-1)
        #print('prob',prob.shape)
        
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob
