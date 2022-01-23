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
        self.tokenizer = BertTokenizer(vocab_file="./BERT/chinese_roberta_wwm_ext_pytorch/vocab.txt")  # 初始化分词器
        self.bert = BertModel.from_pretrained(bert_path)
        self.lstm = nn.LSTM(768, config.hidden_size // 2, config.num_layer,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
      
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)


    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        '''
        t1 = time.time()
        embed = self.word_embed(input_ids)
        # print('embed',embed.shape)
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True)
        # print('packed_inputs',packed_inputs)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        # print('packed_rnn_out',packed_rnn_out)
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        # print('rnn_out',rnn_out.shape)
        hiddens = self.dropout_layer(rnn_out)
        # print('hiddens',hiddens.shape)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        t2 = time.time()
        print('time baseline: ',t2 - t1)
        '''

        t3 = time.time()

        # with BERT
        utts = batch.utt
        # print('utts',utts[2])
        # print('tag_ids',tag_ids[2])
        # print('tag_mask',tag_mask[2])
        # print('lengths',lengths)

        pad_size = 30      # 也称为 max_len (前期统计分析，文本长度最大值为38，取32即可覆盖99%)
        # 补 tag mask 第一个对应[CLS], 第二个对应[SEP] padding 到 pad_size
        
        # 得到BERT标签
        my_tag_ids = []
        my_tag_mask = []
        tag_ids_ = tag_ids.tolist()
        tag_mask_ = tag_mask.tolist()

        # print('tag_mask', tag_mask_[2])
        for i in range(len(lengths)):
        	if (len(tag_ids_[i])+2) < pad_size:
        		tag_id = [0] + tag_ids_[i]  + [0]*(pad_size - len(tag_ids_[i]) - 1)
        		tag_mask_one = [0] + tag_mask_[i] + [0] * (pad_size - len(tag_ids_[i]) - 1)
        	else:
        		tag_id = [0] + tag_ids_[i][:pad_size-1]
        		tag_mask_one = [0] + tag_mask_[i][:pad_size-1]	
        	my_tag_ids.append(tag_id)
        	my_tag_mask.append(tag_mask_one)

        my_tag_ids = torch.LongTensor([_ for _ in my_tag_ids])#.to(self.config.device)
        my_tag_mask = torch.LongTensor([_ for _ in my_tag_mask])#.to(self.config.device)
        # print('my_tag_ids',my_tag_ids.shape,my_tag_ids[2])
        # print('my_tag_mask',my_tag_mask.shape,my_tag_mask[2])


        # 得到BERT输入
        my_input_ids = []     # input char ids
        input_masks = []   # attention mask
        for i in range(len(lengths)):
        	x0 = self.tokenizer.tokenize(utts[i])
        	tokens = ["[CLS]"] + x0 + ["[SEP]"]
        	ids = self.tokenizer.convert_tokens_to_ids(tokens)
        	masks = [1] * len(ids)
        	# 短则补齐，长则切断
        	if len(ids) < pad_size:
        	    masks = masks + [0] * (pad_size - len(ids))
        	    ids = ids + [0] * (pad_size - len(ids))
        	else:
        		masks = masks[:pad_size]
        		ids = ids[:pad_size]
        	my_input_ids.append(ids)
        	input_masks.append(masks)
        
        # print('input_ids',input_ids[2])
        # print('my_input_ids type',type(my_input_ids) )
        # print('my_input_ids shape',len(my_input_ids))
        # print('my_input_ids',my_input_ids[2])

        context = torch.LongTensor([_ for _ in my_input_ids])#.to(self.config.device)
        mask = torch.LongTensor([_ for _ in input_masks])#.to(self.config.device)
        # print('context',context.shape)
        # print('mask',mask.shape)
        
        # t5 = time.time()
        outputs = self.bert(context, attention_mask=mask )
        # t6 = time.time()
        # print('time BERT: ',t6 - t5)

        last_hidden_states = outputs.last_hidden_state
        # print('last_hidden_states',last_hidden_states[0])
        # print('last_hidden_states type',type(last_hidden_states) )
        
        lstm_out, _ = self.lstm(last_hidden_states)
        # print('lstm_out',lstm_out.shape)

        my_hiddens = self.dropout_layer(lstm_out)
        my_tag_output = self.output_layer(my_hiddens, my_tag_mask, my_tag_ids)
        print('my_tag_output',my_tag_output)
         
        t4 = time.time()
        print('time with BERT: ',t4 - t3)

        return my_tag_output

        # return tag_output


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




    def forward_test(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths
        # with BERT
        utts = batch.utt

        pad_size = 30      # 也称为 max_len (前期统计分析，文本长度最大值为38，取32即可覆盖99%)
        # 补 tag mask 第一个对应[CLS], 第二个对应[SEP] padding 到 pad_size
        
        my_tag_ids = []
        my_tag_mask = []

        tag_mask_ = [0]*len(lengths)
        for i in range(len(lengths)):
        	tag_mask_[i] = [1] * lengths[i]
        print('tag_mask_ ' ,tag_mask_[2])

        for i in range(len(lengths)):
        	if (len(tag_mask_[i])+2) < pad_size:
        		tag_mask_one = [0] + tag_mask_[i] + [0] * (pad_size - len(tag_mask_[i]) - 1)
        	else:
        		tag_mask_one = [0] + tag_mask_[i][:pad_size-1]	
        	my_tag_mask.append(tag_mask_one)

        my_tag_mask = torch.LongTensor([_ for _ in my_tag_mask]) #.to(self.device)
        # print('my_tag_ids',my_tag_ids.shape,my_tag_ids[2])
        # print('my_tag_mask',my_tag_mask.shape,my_tag_mask[2])


        # 得到BERT输入
        my_input_ids = []     # input char ids
        input_masks = []   # attention mask
        for i in range(len(lengths)):
        	x0 = self.tokenizer.tokenize(utts[i])
        	tokens = ["[CLS]"] + x0 + ["[SEP]"]
        	ids = self.tokenizer.convert_tokens_to_ids(tokens)
        	masks = [1] * len(ids)
        	# 短则补齐，长则切断
        	if len(ids) < pad_size:
        	    masks = masks + [0] * (pad_size - len(ids))
        	    ids = ids + [0] * (pad_size - len(ids))
        	else:
        		masks = masks[:pad_size]
        		ids = ids[:pad_size]
        	my_input_ids.append(ids)
        	input_masks.append(masks)
        
        # print('input_ids',input_ids[2])
        # print('my_input_ids type',type(my_input_ids) )
        # print('my_input_ids shape',len(my_input_ids))
        # print('my_input_ids',my_input_ids[2])

        context = torch.LongTensor([_ for _ in my_input_ids]) #.to(self.device)
        mask = torch.LongTensor([_ for _ in input_masks]) #.to(self.device)
        # print('context',context.shape)
        # print('mask',mask.shape)
        
        outputs = self.bert(context, attention_mask=mask )

        last_hidden_states = outputs.last_hidden_state
        # print('last_hidden_states',last_hidden_states[0])
        # print('last_hidden_states type',type(last_hidden_states) )
        
        lstm_out, _ = self.lstm(last_hidden_states)
        # print('lstm_out',lstm_out.shape)

        my_hiddens = self.dropout_layer(lstm_out)
        
        my_tag_output = self.output_layer(my_hiddens, my_tag_mask)
        # print('my_tag_output',my_tag_output)
         

        return my_tag_output

        # return tag_output
        

    def decode_test(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob = self.forward_test(batch)
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
        return predictions, labels



class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        # print('--------output_layer input_size:',input_size)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        # print('--------output_layer logits:',logits.shape)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob
