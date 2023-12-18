import pandas as pd
import re

def clean(txt):
    txt = txt.replace('\r', '').replace('\n', ' ')
    txt = re.sub(r'\(.*?\)', r'', txt)
    txt = re.sub(r'（.*?）', r'', txt)
    txt = re.sub(r'【.*?】', r'', txt)
    txt = re.sub(r'([0-9]*)年', r'', txt)
    txt = re.sub(r'([0-9]*)月', r'', txt)
    txt = re.sub(r'([0-9]*)日', r'某日', txt)
    txt = re.sub(r'[“”]', r'', txt)
    txt = re.sub("\s\s+" , " ", txt)
    return txt

def read_file(file_name):
    texts, labels = [], []
    all_data = pd.read_csv(file_name).dropna(subset=['label'])
    for data in all_data.itertuples():
        text = data.post_abstract
        label = data.label
        company_name = data.company_name
        text = text.replace(company_name, '公司')
        text = clean(text)
        texts.append(text)
        labels.append(label)
    d = pd.DataFrame()
    d['text'] = texts
    d['label'] = labels
    return d

data = read_file('./cn_data.csv')

# shuffle并划分训练集和测试集
data = data.sample(frac=1).reset_index(drop=True)
thre = int(0.8*len(data))
sent_train = data[:thre].reset_index(drop=True)
sent_valid = data[thre:].reset_index(drop=True)
sent_train.to_csv('./sent_train.csv', index=False)
sent_valid.to_csv('./sent_valid.csv', index=False)