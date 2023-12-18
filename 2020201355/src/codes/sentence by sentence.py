from hanlp_restful import HanLPClient
import requests
import pandas as pd
import numpy as np
import re
import time



HanLP = HanLPClient('https://www.hanlp.com/api', auth='MzA5OUBiYnMuaGFubHAuY29tOlg0ajkxMkliRkNQRXZhZkE=', language='zh') # auth不填则匿名，zh中文，mul多语种

file_name='HanLP麦当劳(苏州桥店).xlsx'
df=pd.read_excel(file_name,header=0)
class_=[]#961个字好像会超
for i in df['average score']:
    if i>=4:#阈值设的4
        class_.append('好评')
    else:
        class_.append('差评')
class_col = pd.DataFrame(class_, columns=["class"])
df = pd.concat([df,class_col], axis=1)
print(df)
df.to_excel(file_name,index=False)


num=0
core_sentence=[]
for i in df['Comments']:
    print(i)
    i=i.strip()
    tmp_score=[]
    sentences = re.split('[。…!！？?~～]', i)
    sentences = [s.strip() for s in sentences if s.strip()]  # 过滤掉长度为0的句子
    for sentence in sentences:
        try:
            tmp = HanLP.sentiment_analysis(sentence)
            print(tmp)
            tmp_score.append(tmp)
            time.sleep(1)
        except requests.exceptions.ReadTimeout as e:
            print(f"Read timeout error: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
    if df['class'].iloc[num]=='好评':
        max_score = max(tmp_score)
        index_of_max_score = tmp_score.index(max_score)
        core = sentences[index_of_max_score]
        if len(tmp_score) >= 2:
            while len(core)<10 and len(tmp_score)>=2:
                tmp_score.remove(max_score)
                sentences.remove(core)
                max_score = max(tmp_score)
                index_of_max_score = tmp_score.index(max_score)
                core = sentences[index_of_max_score]
        print(max_score,core)
    if df['class'].iloc[num]=='差评':
        min_score = min(tmp_score)
        index_of_min_score = tmp_score.index(min_score)
        core = sentences[index_of_min_score]
        if len(tmp_score) >= 2:
            while (len(core) < 10):
                tmp_score.remove(min_score)
                sentences.remove(core)
                min_score = min(tmp_score)
                index_of_min_score = tmp_score.index(min_score)
                core = sentences[index_of_min_score]
        print(min_score, core)
    core_sentence.append(core)
    num=num+1


core_s = pd.DataFrame(core_sentence, columns=["core sentence"])
result = pd.concat([df,core_s], axis=1)
result.to_excel('test.xlsx',index=False)

