

from transformers import BertTokenizer,BertModel

import torch

bert_path = "./BERT/chinese_roberta_wwm_ext_pytorch"
tokenizer = BertTokenizer(vocab_file="./BERT/chinese_roberta_wwm_ext_pytorch/vocab.txt")  # 初始化分词器

bert = BertModel.from_pretrained(bert_path)
# 修改配置
# model_config.output_hidden_states = True
# model_config.output_attentions = True
# 通过配置和路径导入模型
# bert = BertModel.from_pretrained(bert_path, config = model_config)

x0 = "我爱你中国"
x1 = tokenizer.tokenize(x0)
tokens = ["[CLS]"] + x1 + ["[SEP]"]
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

input_ids = []     # input char ids
input_types = []   # segment ids
input_masks = []   # attention mask
label = []         # 标签
pad_size = 32      # 也称为 max_len (前期统计分析，文本长度最大值为38，取32即可覆盖99%)
types = [0] * len(ids)
masks = [1] * len(ids)
# 短则补齐，长则切断
if len(ids) < pad_size:
    types = types + [1] * (pad_size - len(ids))  # mask部分 segment置为1
    masks = masks + [0] * (pad_size - len(ids))
    ids = ids + [0] * (pad_size - len(ids))
else:
    types = types[:pad_size]
    masks = masks[:pad_size]
    ids = ids[:pad_size]

input_ids.append(ids)
input_types.append(types)
input_masks.append(masks)
assert len(ids) == len(masks) == len(types) == pad_size

y = 1
label.append([int(y)])

print('input_ids',input_ids)
print('input_types',input_types)
print('input_masks',input_masks)
print('label',label)
  
types = input_types
mask = input_masks  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

batch_data = torch.Tensor(input_ids).long().view((1,-1))
print('batch_data',batch_data)
out,_ = bert(batch_data)
print(out,_)



outputs = bert(batch_data,config.output_hidden_states=True)
last_hidden_states = outputs.last_hidden_state
pooler_output = outputs.pooler_output
#hidden_states = outputs.hidden_states
#attentions = outputs.attentions

print(last_hidden_states.shape)
print(pooler_output.shape)
#print(hidden_states.shape)
#print(attentions.shape)





