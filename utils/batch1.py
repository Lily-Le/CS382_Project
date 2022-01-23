#-*- coding:utf-8 -*-
import torch
        

def from_example_list(args, tokenizer, ex_list, device='cpu', train=True, test = False):
    if(not test):
        ex_list = sorted(ex_list, key=lambda x: len(x.input_idx), reverse=True)
    batch = Batch(ex_list, device)
    pad_idx = args.pad_idx
    tag_pad_idx = args.tag_pad_idx

    batch.utt = [ex.utt for ex in ex_list]
    input_lens = [len(ex.input_idx) for ex in ex_list]
    max_len = max(input_lens)
    input_ids = [ex.input_idx + [pad_idx] * (max_len - len(ex.input_idx)) for ex in ex_list]
    batch.lengths = input_lens   

    pad_size = 30      # 也称为 max_len (前期统计分析，文本长度最大值为38，取32即可覆盖99%)

    # 得到BERT输入
    my_input_ids = []     # input char ids
    input_masks = []      # attention mask
    for i in range(len(batch.lengths)):
        x0 = tokenizer.tokenize(batch.utt[i])
        tokens = ["[CLS]"] + x0 + ["[SEP]"]
        ids = tokenizer.convert_tokens_to_ids(tokens)
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

    batch.input_ids = torch.tensor(my_input_ids, dtype=torch.long, device=device)
    batch.input_masks = torch.tensor(input_masks, dtype=torch.long, device=device)

    if train:
        batch.labels = [ex.slotvalue for ex in ex_list]
        tag_lens = [len(ex.tag_id) for ex in ex_list]
        max_tag_lens = max(tag_lens)
        tag_ids = [ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
        tag_mask = [[1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
        
        # 得到BERT标签
        my_tag_ids = []
        my_tag_mask = []

        # print('tag_mask', tag_mask_[2])
        # print('len',len(tag_ids[0]))
        for i in range(len(batch.lengths)):
            if (len(tag_ids[i])) < pad_size:
                #tag_id = [0] + tag_ids[i]  + [0]*(pad_size - len(tag_ids[i]) - 1)
                #tag_mask_one = [0] + tag_mask[i] + [0] * (pad_size - len(tag_ids[i]) - 1)
                tag_id = tag_ids[i]  + [0]*(pad_size - len(tag_ids[i]) )
                tag_mask_one = tag_mask[i] + [0] * (pad_size - len(tag_ids[i]))

            else:
                #tag_id = [0] + tag_ids[i][:pad_size-1]
                #tag_mask_one = [0] + tag_mask[i][:pad_size-1]
                tag_id = tag_ids[i][:pad_size]
                tag_mask_one = tag_mask[i][:pad_size]   
            my_tag_ids.append(tag_id)
            my_tag_mask.append(tag_mask_one)

        batch.tag_ids = torch.tensor(my_tag_ids, dtype=torch.long, device=device)
        batch.tag_mask = torch.tensor(my_tag_mask, dtype=torch.float, device=device)
        '''
        print('batch.utt ',batch.utt[2])
        print('input_ids ',input_ids[2])
        print('my_input_ids ',batch.input_ids[2])
        print('input_masks ',batch.input_masks[2])
        print('tag_ids ',tag_ids[2])
        print('my_tag_ids ',batch.tag_ids[2])
        print('tag_mask ',tag_mask[2])
        print('my_tag_mask ',batch.tag_mask[2])
        '''

    else:
        batch.labels = None
        batch.tag_ids = None
        batch.tag_mask = None

    return batch


class Batch():

    def __init__(self, examples, device):
        super(Batch, self).__init__()

        self.examples = examples
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]