import pandas as pd
import numpy as np
import re

def clean_text(text):#去掉评论里图片表情等乱码
    cleaned_text = re.sub(r'(&#x0A;|<img class="emoji-img"[\s\S]*?alt=""/>)', '', text)
    cleaned_text = cleaned_text.replace('「', '')
    cleaned_text = cleaned_text.replace('」', '')
    cleaned_text = cleaned_text.replace('&#x20;', '')
    cleaned_text = cleaned_text.replace('[薄荷]', '')
    cleaned_text = cleaned_text.replace('[', '')
    cleaned_text = cleaned_text.replace(']', '')
    cleaned_text = cleaned_text.replace('】', '')
    cleaned_text = cleaned_text.replace('【', '')
    return cleaned_text

def limit_text_length(text, max_length=960):#限制文本长度，避免HanLP用不了
    if len(text) > max_length:
        text = text[:max_length]
    return text

#读入
with open('mdl_10.txt', 'r',encoding='utf-8') as file:#换餐厅只需要改这里
    # 读取文件的内容并打印
    content = file.read()

clean_comments=[]
comment_pattern = r'<div class="review-words Hide">(.*?)<div class="less-words">'
# comments = re.findall(comment_pattern, content,re.DOTALL)
short_pattern= r'<div class="review-words">(.*?)</div>'
result1 = re.findall(comment_pattern, content, re.DOTALL)
result2 = re.findall(short_pattern, content, re.DOTALL)

comments = result1 + result2
for comment in comments:
    tmp = comment.strip()
    t1=clean_text(tmp)
    t2=limit_text_length(t1)
    clean_comments.append(t2)
    #print(tmp)
df = pd.DataFrame(clean_comments, columns=["Comments"])
# df.to_excel('comments.xlsx',index=False)

long_score_pattern=r'<span class="score">(.*?)<div class="review-words Hide">'
short_score_pattern=r'<span class="score">(.*?)<div class="review-words">'
long_score= re.findall(long_score_pattern, content, re.DOTALL)
short_score=re.findall(short_score_pattern, content, re.DOTALL)
scores=long_score+short_score
print(len(scores))
#
comment_score=[]
comment_taste=[]
comment_environment=[]
comment_service=[]
#
for i in range(len(scores)):
    tmp = scores[i].strip()
    comment_score.append(tmp)
# print(comment_score)
for score in comment_score:
    t_pattern = r'口味：(\d+\.\d+)'
    e_pattern = r'环境：(\d+\.\d+)'
    s_pattern = r'服务：(\d+\.\d+)'
    t1=re.finditer(t_pattern, score)
    last_match = None
    for match in t1:
        last_match = match.group(1)
    comment_taste.append(last_match)
    e1=re.finditer(e_pattern, score)
    for match in e1:
        last_match = match.group(1)
    comment_environment.append(last_match)
    s1=re.finditer(s_pattern, score)
    for match in s1:
        last_match = match.group(1)
    comment_service.append(last_match)


df1 = pd.DataFrame(comment_taste, columns=["taste score"])
df2 = pd.DataFrame(comment_environment, columns=["environment score"])
df3 = pd.DataFrame(comment_service, columns=["service score"])
result = pd.concat([df,df1, df2,df3], axis=1)
result['taste score'] = pd.to_numeric(result['taste score'], errors='coerce')
result['environment score'] = pd.to_numeric(result['environment score'], errors='coerce')
result['service score'] = pd.to_numeric(result['service score'], errors='coerce')

result['average score']=(result['taste score']+result['environment score']+result['service score'])/3



#总分表
title_pattern=r'<title>“(.*?)”的全部点评'
match=re.search(title_pattern,content)
title=match.group(1)
re_name=str(title)+'.xlsx'
result.to_excel(re_name,index=False)
taste_pattern=r'<span class="item">口味：(.*?)</span>'
environment_pattern=r'<span class="item">环境：(.*?)</span>'
service_pattern=r'<span class="item">服务：(.*?)</span>'
match=re.search(taste_pattern,content)
taste=match.group(1)
match=re.search(environment_pattern,content)
environment=match.group(1)
match=re.search(service_pattern,content)
service=match.group(1)
print(title,taste,environment,service)
total_score=pd.read_excel('餐厅总评分.xlsx',header=0)
new_row = pd.DataFrame({'餐厅名': [title], '口味': [taste], '环境': [environment], '服务': [service]})
total_score = pd.concat([total_score, new_row], ignore_index=True)
total_score.to_excel('餐厅总评分.xlsx',index=False)
file.close()