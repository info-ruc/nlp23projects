from collections import Counter
import pandas as pd
import numpy as np
import json


def get_top_words(words, num, words_to_exclude):
    filtered_words = [word for word in words if word not in words_to_exclude]
    word_counter = Counter(filtered_words)
    top_words = [word for word, _ in word_counter.most_common(num)]
    return top_words

def split_and_clean(text):
    words = text.split(',')
    words = [word.strip() for word in words]
    return words

file_name='HanLP麦当劳(苏州桥店).xlsx'
# 示例用法
df=pd.read_excel(file_name,header=0)
text=''
df1 = df[df['class'] == '好评']
df2 = df[df['class'] == '差评']
for i in df1['Core Sentence Filtered Keywords']:#好评关键词筛选
    text=text+i
word_list=split_and_clean(text)
print(len(word_list))
words_to_exclude=['这家店','麦当劳','里面','这家','周一','儿子','花朵','用餐过程']
top_words = get_top_words(word_list,30,words_to_exclude)
print('好评',top_words)
text=''
for i in df2['Core Sentence Filtered Keywords']:#差评关键词筛选
    text=text+i
word_list=split_and_clean(text)
print(len(word_list))
top_words = get_top_words(word_list,50,words_to_exclude)
print('差评',top_words)



#=====================================================可视化，画词云图==============================================================
from wordcloud import WordCloud
import matplotlib.pyplot as plt

good_list = df1['Comments'].tolist()
# 将文本列表合并成一个字符串
text = " ".join(good_list)
font_path = "C:\Windows\Fonts\simsun.ttc" # 宋体路径
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体，例如 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号
wordcloud = WordCloud(width=800, height=400, background_color="white", font_path=font_path).generate(text)

# 绘制词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("麦当劳好评词云图", fontsize=16)
plt.savefig("麦当劳好评词云图.png", bbox_inches='tight')
plt.show()

bad_list = df2['Comments'].tolist()
# 将文本列表合并成一个字符串
text = " ".join(bad_list)
wordcloud = WordCloud(width=800, height=400, background_color="white", font_path=font_path).generate(text)
# 绘制词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("麦当劳差评词云图", fontsize=16)
plt.savefig("麦当劳差评词云图.png", bbox_inches='tight')
plt.show()

core_good_list = df1['core sentence'].tolist()
# 将文本列表合并成一个字符串
text = " ".join(core_good_list)
wordcloud = WordCloud(width=800, height=400, background_color="white", font_path=font_path).generate(text)
# 绘制词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("麦当劳核心好评词云图", fontsize=16)
plt.savefig("麦当劳核心好评词云图.png", bbox_inches='tight')
plt.show()

core_bad_list = df2['core sentence'].tolist()
# 将文本列表合并成一个字符串
text = " ".join(core_bad_list)
wordcloud = WordCloud(width=800, height=400, background_color="white", font_path=font_path).generate(text)
# 绘制词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("麦当劳核心差评词云图", fontsize=16)
plt.savefig("麦当劳核心差评词云图.png", bbox_inches='tight')
plt.show()
