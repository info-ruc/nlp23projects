from hanlp_restful import HanLPClient
import requests
import pandas as pd
import numpy as np
import re
import time



HanLP = HanLPClient('https://www.hanlp.com/api', auth='MzA5OUBiYnMuaGFubHAuY29tOlg0ajkxMkliRkNQRXZhZkE=', language='zh') # auth不填则匿名，zh中文，mul多语种
file_name='HanLP麦当劳(苏州桥店).xlsx'
df=pd.read_excel('麦当劳(苏州桥店).xlsx',header=0)
HanLP_score=[]#961个字好像会超

for i in df['Comments']:
    print(i)

    try:
        tmp=HanLP.sentiment_analysis(i)
        print(tmp)
        HanLP_score.append(tmp)
        time.sleep(1)
    except requests.exceptions.ReadTimeout as e:
        print(f"Read timeout error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")

score = pd.DataFrame(HanLP_score, columns=["HanLP score"])
result = pd.concat([df,score], axis=1)
result.to_excel(file_name,index=False)

def map_to_0_5_range(original_value):#HanLP评分折到大众点评评分
    mapped_value = (original_value + 1) * 2.5
    return mapped_value

df=pd.read_excel(file_name,header=0)
mapped_score=[]
for i in df["HanLP score"]:
    tmp=map_to_0_5_range(i)
    mapped_score.append(tmp)
score = pd.DataFrame(mapped_score, columns=["mapped score"])
result = pd.concat([df,score], axis=1)
result.to_excel(file_name,index=False)


df=pd.read_excel(file_name,header=0)
key_words=[]#961个字好像会超

for i in df['Comments']:
    print(i)
    tmp=HanLP.keyphrase_extraction(i)
    print(tmp)
    key_words.append(str(tmp))
    time.sleep(1)
kwords = pd.DataFrame(key_words, columns=["Key Words"])
result = pd.concat([df,kwords], axis=1)
result.to_excel(file_name,index=False)
