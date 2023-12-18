import os
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
import pandas as pd
import re
import string

# Remove punctuations, links, mentions and \r\n new line characters
def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list = string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    banned_list = banned_list.replace('$', '')
    banned_list = banned_list.replace('%', '')
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

# Clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove hashtags symbol from words in the middle of the sentence
    return new_tweet2

def remove_mult_spaces(text): # remove multiple spaces
    return re.sub("\s\s+" , " ", text)

def read_data(file, us=True):
    all_data = pd.read_csv(file, encoding='utf-8')
    texts, labels, max_lengths = [], [], []
    for data in all_data.itertuples():
        text = data.text
        label = data.label
        if us:
            text = remove_mult_spaces(clean_hashtags(strip_all_entities(text)))
        max_lengths.append(len(text))
        texts.append(text)
        labels.append(label)
    if os.path.split(file)[-1] == "sent_train.csv":
        max_len = max(max_lengths)
        return texts, labels, max_len
    return texts, labels, 0

class MyDataset(Dataset):
    def __init__(self, texts, labels, args):
        self.all_text = texts
        self.all_label = labels
        self.max_len = args['max_len']
        self.tokenizer = BertTokenizer.from_pretrained(args['bert_model'])

    def __getitem__(self, index):
        # 取出一条数据
        text = self.all_text[index][:self.max_len]
        label = self.all_label[index]

        text_id = self.tokenizer.tokenize(text)
        text_id = ["[CLS]"] + text_id

        # 编码
        token_id = self.tokenizer.convert_tokens_to_ids(text_id)
        mask = [1] * len(token_id) + [0] * (self.max_len + 2 - len(token_id))
        token_ids = token_id + [0] * (self.max_len + 2 - len(token_id))
        label = int(label)

        token_ids = torch.tensor(token_ids)
        mask = torch.tensor(mask)
        label = torch.tensor(label)

        return (token_ids, mask), label

    def __len__(self):
        return len(self.all_text)
