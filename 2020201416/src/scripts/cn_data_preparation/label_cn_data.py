import datetime
import pandas as pd
import tqdm
import tushare as ts

ts.set_token('$Your-Tushare-Token$')
api = ts.pro_api()

columns = [
    'title', 'company_name', 'post_content', 'post_abstract', 'date', 'label'
]

def add_label(x, df_price, foward_days = 5, threshold = 0.02):
    publish_date = x.publish_date.strftime("%Y%m%d")
    last_date = df_price[df_price.trade_date < publish_date].iloc[-1].name
    this_date_index = last_date + 1
    next_date_index = this_date_index + foward_days
    
    if next_date_index > df_price.shape[0]-1:
        next_date_index = df_price.shape[0]-1

    this = df_price[df_price.index == this_date_index].open.values[0]
    next_ = df_price[df_price.index == next_date_index].open.values[0]
    change = (next_ - this) / this
    if change < -threshold:
        return 0
    elif change > threshold:
        return 1
    else:
        return 2

def process_label(df, foward_days = 5, threshold = 0.02):
    # 获取日期
    df["post_publish_time"] = pd.to_datetime(df["post_publish_time"])
    df["date"] = df["post_publish_time"].dt.date
    df["time"] = df["post_publish_time"].dt.time
    df["hour"] = df["post_publish_time"].dt.hour

    # 确定时间窗
    start_date = df["date"].min() - datetime.timedelta(days = 10)
    end_date = df["date"].max() + datetime.timedelta(days = 25)
    start_date = start_date.strftime("%Y%m%d")
    end_date = end_date.strftime("%Y%m%d")
    code = df.code_name.unique()[0]

    # 查询股价信息
    df_price = api.query('daily', ts_code=code, start_date=start_date, end_date=end_date)
    df_price = df_price.sort_values(by=['trade_date']).reset_index(drop=True)

    # 标注
    df['publish_date'] = df.apply(lambda x : x['date'] if x['hour'] <15 else x['date'] + datetime.timedelta(days = 1), axis=1)
    tqdm.tqdm.pandas(desc='apply')
    df["label"] = df.progress_apply(lambda x : add_label(x, df_price=df_price, foward_days=foward_days, threshold=threshold), axis=1)
    company_name = api.query('namechange', ts_code=code).loc[0, 'name']
    company_name = company_name.split('-')[0] # 删除可能的其他信息（例：奇安信-U）
    df['company_name'] = company_name
    df = df[columns]
    return df

data = pd.read_csv('./cn_data.csv')
new_data = pd.DataFrame()
stock_list = list(set(data['code_name'].tolist()))
stock_list.sort()
for stock in stock_list :
    this_data = data[data['code_name'] == stock].reset_index(drop=True)
    this_data = process_label(this_data)
    new_data = pd.concat([new_data, this_data], axis=0, ignore_index=True)
new_data.to_csv('./cn_data.csv', index=False)