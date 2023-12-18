from hanlp_restful import HanLPClient
import requests
import pandas as pd
import numpy as np
import re
import time

def clean_text(text):#关键词格式清洗
    cleaned_text = text.replace('{', '')
    cleaned_text = cleaned_text.replace('}', '')
    return cleaned_text

def filter_text_by_threshold(input_str, threshold=0.3):#阈值可以自己调，和主题相关的程度
    try:
        # 将输入的字符串转换为字典
        data_dict = eval("{" + input_str + "}")
        # 过滤出值大于等于阈值的项，只保留键
        filtered_keys = [key for key, value in data_dict.items() if value >= threshold]
        # 将结果拼接成字符串
        result_str = ', '.join(filtered_keys)
        return result_str
    except Exception as e:
        return f"Error: {e}"


HanLP = HanLPClient('https://www.hanlp.com/api', auth='MzA5OUBiYnMuaGFubHAuY29tOlg0ajkxMkliRkNQRXZhZkE=', language='zh') # auth不填则匿名，zh中文，mul多语种

df=pd.read_excel('test.xlsx',header=0)
key_words=[]#961个字好像会超
file_name='HanLP麦当劳(苏州桥店).xlsx'
for i in df['core sentence']:
    print(i)
    tmp=HanLP.keyphrase_extraction(i)
    print(tmp)
    key_words.append(str(tmp))
    time.sleep(1)
kwords = pd.DataFrame(key_words, columns=["Core Sentence Keywords"])
result = pd.concat([df,kwords], axis=1)
result.to_excel(file_name,index=False)

keyword_clean=[]
df=pd.read_excel(file_name,header=0)
for i in df['Core Sentence Keywords']:
    clean = clean_text(i)
    filtered = filter_text_by_threshold(clean)
    keyword_clean.append(filtered)

k = pd.DataFrame(keyword_clean, columns=["Core Sentence Filtered Keywords"])
result = pd.concat([df,k], axis=1)
result.to_excel(file_name,index=False)