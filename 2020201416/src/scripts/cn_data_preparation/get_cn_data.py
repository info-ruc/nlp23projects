# git clone https://github.com/AI4Finance-Foundation/FinNLP
import os
import tqdm
import pandas as pd
import requests
from lxml import etree
import json
from time import sleep
from FinNLP.finnlp.data_sources.news.eastmoney_streaming import Eastmoney_Streaming

df = pd.read_csv("hs_300.csv") # 沪深300数据
stock_list = df.SECURITY_CODE.unique()
stock_list = [str(s).zfill(6) for s in stock_list]

result_path = "../results"
link_base = "https://guba.eastmoney.com"

# 在股票首页获取所有资讯的网页链接
def get_news_link(stock):
    print(f"Collecting {stock}")
    config = {
        "max_retry": 5,
    }

    result_path = "./results"
    result_path = os.path.join(result_path, f"{stock}.csv")
    downloader = Eastmoney_Streaming(config)
    downloader.download_streaming_stock(stock, rounds = 0)
    return downloader.dataframe

# 资讯信息列
columns = [
    'post_user', 'post_guba', 'post_publish_time', 'post_last_time',
    'post_display_time', 'post_checkState', 'post_click_count',
    'post_forward_count', 'post_comment_count', 'post_comment_authority',
    'post_like_count', 'post_is_like', 'post_is_collected', 'post_type',
    'post_source_id', 'post_top_status', 'post_status', 'post_from',
    'post_from_num', 'post_pdf_url', 'code_name', 'ask_chairman_state',
    'extend', 'relate_topic', 'source_extend', 'digest_type',
    'reptile_state', 'extend_version', 'post_ip_address','post_id',
    'post_title', 'post_content', 'post_abstract', 'post_state'
]

# 获取一条资讯
def get_one_content(x):
    url = link_base + x
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Referer": link_base + "/",
    }

    requests.DEFAULT_RETRIES = 5
    s = requests.session()
    s.keep_alive = False
    
    while 1:
        try:
            response = requests.get(url=url, headers=headers)
            response.encoding = 'utf-8'
            if response.status_code == 200:
                res = etree.HTML(response.text)
                res = res.xpath("//script[2]//text()")[0]
                res = json.loads(res[17:])
                res = [res[key] for key in columns]
                sleep(8) # 防范反爬虫
                return res
        except:
            pass

# 获取资讯内容
data = pd.DataFrame()
for stock in stock_list:
    this_data = get_news_link(stock)
    tqdm.tqdm.pandas(desc='apply')
    this_data[columns] = this_data.progress_apply(lambda x : get_one_content(x['content link']), axis=1, result_type="expand")
    data = pd.concat([data, this_data], axis=0, ignore_index=True)
data.to_csv('./cn_data.csv', index=False)