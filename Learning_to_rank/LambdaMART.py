import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb

def LambdaMART(data_modeling):
    encoder = LabelEncoder()
    data = data_modeling.copy()
    #
    data= data.sort_values(by='timestamp')
    # Chuyển tên cột
    data = data.rename(columns={"timestamp": "Date", "symbol": "Asset"})
    # Đặt asset_code đưa vào train test val
    data['Asset_Code'] = encoder.fit_transform(data['Asset'])
    asset_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    data = data.sort_values(by=['Date', 'Asset_Code'])
    # Đặt asset code thành index
    data.set_index('Asset_Code',inplace=True)
    label_encoder = LabelEncoder()
    # label encoder qid thành số
    data['qid'] = label_encoder.fit_transform(data['Date'])
    # tạo biến lợi nhuận tương lai thành 10 nhóm
    data['future_return_rs'] = (
        data.groupby('Date')['future_return']
        .apply(lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop'))
        .reset_index(drop=True)
    )
    train = data[(data['Date'] >= '2022-01-01') & (data['Date'] < '2023-03-01')]
    val = data[(data['Date'] >= '2023-03-01') & (data['Date'] < '2023-06-01')]
    test =data[data['Date'] >= '2023-06-01']
    

    X_train = train[['cumulative_return_3m', 'normalized_return_3m',
                    'cumulative_return_6m', 'normalized_return_6m',
                    'cumulative_return_12m', 'normalized_return_12m',
                    'EMA12', 'EMA26','MACD', 'Signal_Line','raw_return',
                    'log_return','open','high','close','low','volume']]

    X_test =test[['cumulative_return_3m', 'normalized_return_3m',
                    'cumulative_return_6m', 'normalized_return_6m',
                    'cumulative_return_12m', 'normalized_return_12m',
                    'EMA12', 'EMA26','MACD', 'Signal_Line','raw_return',
                    'log_return','open','high','close','low','volume']]

    X_val =val[['cumulative_return_3m', 'normalized_return_3m',
                    'cumulative_return_6m', 'normalized_return_6m',
                    'cumulative_return_12m', 'normalized_return_12m',
                    'EMA12', 'EMA26','MACD', 'Signal_Line','raw_return',
                    'log_return','open','high','close','low','volume']]

    y_train = train[['future_return_rs']]
    y_test= test[['future_return_rs']]
    y_val= val[['future_return_rs']]

    label_encoder = LabelEncoder()
    qid_train = train['qid']
    qid_test = test['qid']
    qid_val = val['qid']

    print(X_train.shape)
    print(y_train.shape)
    print(qid_train.shape)
    ranker = xgb.XGBRanker( objective='rank:pairwise', eval_metric='ndcg',
                        tree_method='hist', eta=1, max_depth=8, n_estimators=5, device='gpu' )

    ranker.fit( X=X_train, y=y_train, qid=qid_train, eval_set=[(X_val, y_val)], eval_qid=[qid_val], verbose=False)
    scores = ranker.predict(X_test)
    test['diem'] = scores
    test['Rank'] = test.groupby('Date')['diem'].rank(ascending=False, method='dense')
    # Lấy 10% đứng đầu và 10% đứng cuối theo từng tháng
    data=test
    # Lấy 10% đứng đầu và 10% đứng cuối theo từng tháng
    top_10_percent = data.groupby('Date').apply(lambda x: x.nlargest(int(len(x) * 0.1), 'diem')).reset_index(drop=True)
    bottom_10_percent = data.groupby('Date').apply(lambda x: x.nsmallest(int(len(x) * 0.1), 'diem')).reset_index(drop=True)
    # Tạo DataFrame chứa các tháng, danh sách các asset top 10% và bottom 10%
    result = []
    # Lặp qua từng nhóm ngày và tạo danh sách asset cho top 10% và bottom 10%
    for date in data['Date'].unique():
        top_assets = top_10_percent[top_10_percent['Date'] == date]['Asset'].tolist()
        bottom_assets = bottom_10_percent[bottom_10_percent['Date'] == date]['Asset'].tolist()
        result.append(pd.DataFrame({'Date': [date],
                                    'Top_10%_Assets': [top_assets],
                                    'Bottom_10%_Assets': [bottom_assets]}))

    final_result = pd.concat(result, ignore_index=True)
    final_result.to_csv('LamdaMART_rank.csv')