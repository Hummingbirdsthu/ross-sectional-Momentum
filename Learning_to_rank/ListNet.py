import numpy as np
import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


# Mô hình ListNet
class Model(chainer.Chain):
    def __init__(self, n_in, n_units1, n_units2, n_out):
        super(Model, self).__init__(
            l1=L.Linear(n_in, n_units1),
            l2=L.Linear(n_units1, n_units2),
            l3=L.Linear(n_units2, n_out),
        )

    def __call__(self, x, t):
        h1 = self.l1(x)
        y = self.l3(F.relu(self.l2(F.relu(self.l1(x))))).data
        self.loss = self.jsd(t, y)
        return self.loss

    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h = F.relu(self.l3(h2))
        return h.data

    def kld(self, vec_true, vec_compare):
        ind = vec_true.data * vec_compare.data > 0
        ind_var = chainer.Variable(ind)
        include_nan = vec_true * F.log(vec_true / vec_compare)
        z = chainer.Variable(np.zeros((len(ind), 1), dtype=np.float32))
        return F.sum(F.where(ind_var, include_nan, z))

    def jsd(self, vec_true, vec_compare):
        vec_mean = 0.5 * (vec_true + vec_compare)
        return 0.5 * self.kld(vec_true, vec_mean) + 0.5 * self.kld(vec_compare, vec_mean)
        
    def ndcg(self, y_true, y_score, k=100):
        y_true = y_true.ravel()
        y_score = y_score.ravel()
        y_true_sorted = sorted(y_true, reverse=True)
        ideal_dcg = 0
        for i in range(k):
            ideal_dcg += (2 ** y_true_sorted[i] - 1.) / np.log2(i + 2)
        dcg = 0
        argsort_indices = np.argsort(y_score)[::-1]
        for i in range(k):
            dcg += (2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2)
        ndcg = dcg / ideal_dcg
        return ndcg
    
    
    
# Mã hóa các tài sản
encoder = LabelEncoder()
def data_model(data):
    data= data.sort_values(by='timestamp')
    data = data.rename(columns={"timestamp": "Date", "symbol": "Asset"})
    data['Asset_Code'] = encoder.fit_transform(data['Asset'])
    asset_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    data = data.sort_values(by=['Date', 'Asset_Code'])
    data.set_index('Asset_Code',inplace=True)
    label_encoder = LabelEncoder()
    data['qid'] = label_encoder.fit_transform(data['Date'])
    data['future_return_rs'] = (
        data.groupby('Date')['future_return']
        .apply(lambda x: pd.qcut(x, q=20, labels=False, duplicates='drop'))
        .reset_index(drop=True)
    )
    data = data.groupby(['Date', 'Asset'], as_index=False).first()

    features = ['cumulative_return_3m', 'normalized_return_3m',
                    'cumulative_return_6m', 'normalized_return_6m',
                    'cumulative_return_12m', 'normalized_return_12m',
                    'EMA12', 'EMA26','MACD', 'Signal_Line','open','high',
                    'close','low','volume']

    X = data[features].values
    y = data[['future_return_rs']].values
    # Chuẩn hóa dữ liệu
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)


    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    # Huấn luyện mô hình ListNet
    n_in = X.shape[1]
    n_units1, n_units2, n_out = 64, 32, 1
    model = Model(n_in, n_units1, n_units2, n_out)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # Huấn luyện
    n_epoch = 50
    batchsize = 64
    import numpy as np
    for epoch in range(n_epoch):
        print(f"Epoch {epoch + 1}/{n_epoch}")
        
        perm = np.random.permutation(len(X_train))

        sum_loss = 0
        y_true_all = []
        y_pred_all = []
        
        for i in tqdm(range(0, len(X_train), batchsize)):
            x = chainer.Variable(np.asarray(X_train[perm[i:i + batchsize]], dtype=np.float32))
            t = chainer.Variable(np.asarray(y_train[perm[i:i + batchsize]], dtype=np.float32))
            optimizer.update(model, x, t)
            sum_loss += float(model(x, t).data)
            
            # Dự đoán xếp hạng và lưu trữ true và predicted values
            y_pred = model.predict(x).ravel()
            y_true_all.append(t.data.ravel())  # Sử dụng .data và ravel()
            y_pred_all.append(y_pred)
        
        # Tính toán và in ra chỉ số NDCG
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        
        ndcg_score = model.ndcg(y_true_all, y_pred_all, k=100)
        print(f"Train loss: {sum_loss / len(X_train)}")
        print(f"NDCG at : {ndcg_score}")
    return model,scaler_X
# Hàm tái cân bằng danh mục đầu tư hàng tháng
def ListNet(data, n=10):
    model, scaler_X = data_model(data)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['month'] = data['Date'].dt.to_period('M')  # Chuyển timestamp thành chu kỳ tháng
    portfolio = []

    for month, group in data.groupby('month'):
        # Lấy các cột features đã định nghĩa
        features = ['cumulative_return_3m', 'normalized_return_3m',
                 'cumulative_return_6m', 'normalized_return_6m',
                 'cumulative_return_12m', 'normalized_return_12m',
                'EMA12', 'EMA26','MACD', 'Signal_Line','open','high',
                'close','low','volume']
        
        # Chuẩn bị dữ liệu đầu vào
        X = group[features].values
        X_scaled = scaler_X.transform(X)
        symbols = group['Asset'].values
        returns = group['raw_return'].values

        # Dự đoán xếp hạng (rank) với mô hình
        x_var = chainer.Variable(np.asarray(X_scaled, dtype=np.float32))
        y_pred = model.predict(x_var)
        y_pred_rank = np.argsort(-y_pred, axis=0).ravel() + 1  # Sắp xếp thứ hạng giảm dần

        # Tạo DataFrame lưu kết quả
        results = pd.DataFrame({
            'Symbol': symbols,
            'PredictedRank': y_pred_rank,
            'Returns': returns,
            'Month': str(month)
        })

        # Lấy top n symbol có thứ hạng cao nhất và thấp nhất
        top_n_highest = results.sort_values('PredictedRank').head(n)
        top_n_lowest = results.sort_values('PredictedRank', ascending=False).head(n)

        # Gán giá trị PredictedRank = 1 cho top n cao nhất và -1 cho top n thấp nhất
        top_n_highest['PredictedRank'] = 1
        top_n_lowest['PredictedRank'] = -1

        # Thêm vào danh mục đầu tư
        portfolio.append(top_n_highest)
        portfolio.append(top_n_lowest)
        portfolio.to_csv('ListNet_rank.csv', index=False)
    # Kết hợp tất cả các tháng lại thành một DataFrame
    return pd.concat(portfolio)


