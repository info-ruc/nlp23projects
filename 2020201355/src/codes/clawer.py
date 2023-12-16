import requests
import re
from lxml import etree
import pandas as pd
import time
import random

# headers = {
# 'Cookie':'_lxsdk_cuid=18b411e9ea9c8-09d9da6ce1e04-745d5774-144000-18b411e9eaac8; _lxsdk=18b411e9ea9c8-09d9da6ce1e04-745d5774-144000-18b411e9eaac8; WEBDFPID=7xy9ww672wzz5956y8yxxz57u58969wy81y588vz7ww97958w5uu17ww-2012964608087-1697604606904WUUOOYAfd79fef3d01d5e9aadc18ccd4d0c95074545; _hc.v=f15d6e9e-f00b-0418-528b-d09daf897218.1697604608; ua=%E5%90%91%E6%97%A5%E8%91%B5_926739; ctu=ec89533d9813854dbd52653040cd3bfcae1e6fa7437360fd0c57b67c8540f1aa; fspop=test; cy=2; cye=beijing; Hm_lvt_602b80cf8079ae6591966cc70a3940e7=1700468406; s_ViewType=10; qruuid=b699d451-4cb1-428e-b1ee-7e995afbfd28; dplet=b32439e9af0c3645559073bb59faf5d3; dper=18cb4d6b659d91dd50824393768866342937dce80c6bfaabd11a24eaec82ac71795230984715b017d31097a71449497476d8d0d95a9b8124eb22c9606bf8b534; ll=7fd06e815b796be3df069dec7836c3df; Hm_lpvt_602b80cf8079ae6591966cc70a3940e7=1700469161; _lxsdk_s=18bebd0b338-85-44e-8a8%7C%7C834',
# 'Host':'www.dianping.com',
# #'Referer': 'https://www.dianping.com/shop/l87AcPXlZlCVgjsL/review_all/p10',
# 'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0'
# }
#
# response = requests.get('https://www.dianping.com/shop/G1mcJXJ5fPZb21pG/review_all',headers =headers)#品质伊骊
# text1=response.text
# texts=[]
# for i in range(2,11):#一共取10页，150条
#     url = 'https://www.dianping.com/shop/G1mcJXJ5fPZb21pG/review_all/' + 'p' + str(i)
#     response = requests.get(url, headers=headers)
#     t = response.text
#     texts.append(t)
#
# with open('pzyl_10_1.txt', 'a',encoding='utf-8') as file:#品质伊犁10页
#     file.write(text1)
#     for i in texts:
#         file.write(i)
# file.close()

#麦当劳
headers = {
#'Cookie':'_lxsdk_cuid=18b411e9ea9c8-09d9da6ce1e04-745d5774-144000-18b411e9eaac8; _lxsdk=18b411e9ea9c8-09d9da6ce1e04-745d5774-144000-18b411e9eaac8; WEBDFPID=7xy9ww672wzz5956y8yxxz57u58969wy81y588vz7ww97958w5uu17ww-2012964608087-1697604606904WUUOOYAfd79fef3d01d5e9aadc18ccd4d0c95074545; _hc.v=f15d6e9e-f00b-0418-528b-d09daf897218.1697604608; ua=%E5%90%91%E6%97%A5%E8%91%B5_926739; ctu=ec89533d9813854dbd52653040cd3bfcae1e6fa7437360fd0c57b67c8540f1aa; fspop=test; cy=2; cye=beijing; Hm_lvt_602b80cf8079ae6591966cc70a3940e7=1700468406; s_ViewType=10; qruuid=b699d451-4cb1-428e-b1ee-7e995afbfd28; dplet=b32439e9af0c3645559073bb59faf5d3; dper=18cb4d6b659d91dd50824393768866342937dce80c6bfaabd11a24eaec82ac71795230984715b017d31097a71449497476d8d0d95a9b8124eb22c9606bf8b534; ll=7fd06e815b796be3df069dec7836c3df; Hm_lpvt_602b80cf8079ae6591966cc70a3940e7=1700469161; _lxsdk_s=18bebd0b338-85-44e-8a8%7C%7C834',
'Cookie':'s_ViewType=10; WEBDFPID=7xy9ww672wzz5956y8yxxz57u58969wy81y588vz7ww97958w5uu17ww-2012964608087-1697604606904WUUOOYAfd79fef3d01d5e9aadc18ccd4d0c95074545; _lxsdk_cuid=18b411e9ea9c8-09d9da6ce1e04-745d5774-144000-18b411e9eaac8; _lxsdk=18b411e9ea9c8-09d9da6ce1e04-745d5774-144000-18b411e9eaac8; _hc.v=e1e600a2-abc1-85b0-e11e-13559b107d5d.1700472237; ua=%E5%90%91%E6%97%A5%E8%91%B5_926739; ctu=ec89533d9813854dbd52653040cd3bfc290924d1b2f3d95751afb8375a715443; fspop=test; _lx_utm=utm_source%3Dbing%26utm_medium%3Dorganic; Hm_lvt_602b80cf8079ae6591966cc70a3940e7=1700468406,1701013263,1702696732; qruuid=36db2d5b-0292-48b2-96f2-442709c4e42d; dplet=b170140b726cfafd168c0dbf6a05a871; dper=18cb4d6b659d91dd5082439376886634050a6e40a5cf82b56be5fd3bf8a93bb67aca4af1b28fe730b18c714a208904042edf813291b940d18f553b75a67c97cb; ll=7fd06e815b796be3df069dec7836c3df; cy=2; cye=beijing; Hm_lpvt_602b80cf8079ae6591966cc70a3940e7=1702697650; _lxsdk_s=18c70a2155f-9ed-c78-e1b%7C%7C151',
'Host':'www.dianping.com',
'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
}

response = requests.get('https://www.dianping.com/shop/H4RFfW4GElACalyk/review_all',headers =headers)#麦当劳
text1=response.text
texts=[]
for i in range(2,11):#一共取10页，150条
    url = 'https://www.dianping.com/shop/H4RFfW4GElACalyk/review_all/' + 'p' + str(i)
    response = requests.get(url, headers=headers)
    t = response.text
    texts.append(t)

with open('mdl_10.txt', 'a',encoding='utf-8') as file:#品质伊犁10页
    file.write(text1)
    for i in texts:
        file.write(i)
file.close()

