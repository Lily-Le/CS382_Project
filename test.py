#coding=utf8
import sys, os, time, gc
from torch.optim import Adam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.test_example import Test_Example as Example
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_bert_tagging import SLUTagging
import json
from transformers import BertTokenizer,BertModel
# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
set_random_seed(args.seed)
device = set_torch_device(args.device)

print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
# test_path = os.path.join(args.dataroot, 'train.json')
Example.configuration(args.dataroot, train_path=test_path, word2vec_path=args.word2vec_path)
test_dataset = Example.load_test_dataset(test_path)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: test -> %d " % len(test_dataset) )

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)


model = SLUTagging(args).to(device)
# model = torch.load(PATH)
# Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)
model.load_state_dict(torch.load('./best_model.bin'))
tokenizer = BertTokenizer(vocab_file="BERT/chinese_roberta_wwm_ext_pytorch/vocab.txt")  # 初始化分词器

with open(test_path, 'r') as f:
    test_json=json.load(f)

def reshape_ignoreUttid(test_json):
    examples = []
    for data in test_json:
        for utt in data:
            examples.append(utt)
    return examples

def inverse_reshape(test_result):
    res=[]
    cur_len=0
    for data in test_result:
        if data['utt_id']>1:
            res[cur_len-1].append(data)
        else:
            res.append([data])
            cur_len += 1
    return res

test_result = reshape_ignoreUttid(test_json) #if a dialog has multiple utts, seperate the utts

def decode_test():
    model.eval()
    dataset = test_dataset
    predictions, labels = [], []
    count = 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            # current_batch = from_example_list(args, cur_dataset, device, train = False, test = True)
            current_batch = from_example_list(args, tokenizer, cur_dataset, device, train = False, test = True)
            pred, label = model.decode_test(Example.label_vocab, current_batch)


            with open("./test.json","w") as f:
                for j in range(len(pred)):
                    test_result[i+j]['pred'] = pred[j]
                    # json.dump(pred[i],f,ensure_ascii=False,indent=1)

            # print('pred size ',len(pred))
            # print('pred ',pred[0])
            # print('pred ',pred[1])
            # print('pred ',pred[2])

            predictions.extend(pred)
            count += 1

    torch.cuda.empty_cache()
    gc.collect()


start_time = time.time()
decode_test()
print('Finished!')
final_result = inverse_reshape(test_result)
with open("./test_result.json", "w") as f:
     json.dump(final_result, f, ensure_ascii=False, indent=1)


