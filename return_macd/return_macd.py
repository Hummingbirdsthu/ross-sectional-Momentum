import os
import pandas as pd
import numpy as np
from datetime import datetime

# Giả sử 'train' đã được đọc từ một file nào đó, ví dụ như CSV
# train = pd.read_csv('your_data.csv')
# Thiết lập tham số MACD
params = {
    "macd_span_short": [12, 24, 48],
    "macd_span_long": [26, 52, 104],
    "macd_std_short": 9,
    "macd_std_long": 21
}
periods = [3, 6, 9, 12]
# Hàm tính toán lợi nhuận tích lũy cho từng mã tiền điện tử
def calculate_cumulative_returns(df, period , coin_column):
    df[coin_column]= df[coin_column].pct_change(periods=period).add(1).cumprod().sub(1)
    return df

# Hàm tính toán lợi nhuận tích lũy cho tất cả các mã tiền điện tử và lưu thành các file CSV cho từng kỳ hạn
def calculate_and_save_cumulative_returns(df, periods = periods, prefix=''):
    # Tạo thư mục 'return' nếu chưa tồn tại
    if not os.path.exists('return'):
        os.makedirs('return')

    # Tạo một DataFrame mới với timestamp làm chỉ mục
    result_df = pd.DataFrame()

    # Duyệt qua các cột tiền điện tử trong dataframe
    for period in periods:
        temp_df = pd.DataFrame()  # DataFrame tạm thời để chứa kết quả của từng kỳ hạn
        temp_df['timestamp'] = df['timestamp']  # Thêm cột timestamp vào kết quả

        # Duyệt qua tất cả các mã tiền điện tử và tính toán lợi nhuận tích lũy
        for coin_column in df.columns:
            if coin_column != 'timestamp':  # Bỏ qua cột timestamp
                # Tính lợi nhuận tích lũy cho từng mã tiền điện tử
                df = calculate_cumulative_returns(df, period, coin_column)
                # Thêm cột lợi nhuận tích lũy của mã tiền điện tử vào temp_df
                temp_df[coin_column] = df[coin_column]

        # Lưu kết quả thành file CSV trong thư mục 'return' cho từng kỳ hạn
        temp_df.to_csv(f'data/return/{prefix}_cumulative_returns_{period}m.csv', index=False)
        print(f"Đã lưu file return/{prefix}_cumulative_returns_{period}m.csv")

# Danh sách các kỳ hạn cần tính toán
# Hàm tính MACD
def compute_macd(df_column, short_period, long_period):
    ema_short = df_column.ewm(span=short_period, adjust=False).mean()
    ema_long = df_column.ewm(span=long_period, adjust=False).mean()
    macd = ema_short - ema_long
    return macd

# Hàm tính độ biến động
def compute_volatility(df_column, window=20):
    return df_column.rolling(window=window).std()

# Hàm tính toán tín hiệu MACD chuẩn hóa cho một DataFrame
def compute_macd_signals(df, short_periods, long_periods, window=20):
    macd_signals_df = pd.DataFrame()

    # Tính MACD và chuẩn hóa cho từng mã tiền điện tử
    for Sk, Lk in zip(short_periods, long_periods):
        for column in df.columns:  # Lặp qua các cột trong DataFrame
            if column != 'timestamp':  # Không tính trên cột timestamp
                macd = compute_macd(df[column], Sk, Lk)
                volatility = compute_volatility(df[column], window)
                normalized_macd = macd / volatility

                # Tạo tên cột tín hiệu
                signal_name = f'MACD_{Sk}_{Lk}_{column}'

                # Thêm tín hiệu vào DataFrame
                macd_signals_df[signal_name] = normalized_macd

    # Đảm bảo rằng 'timestamp' là kiểu dữ liệu datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Thêm cột 'timestamp' vào DataFrame chứa các tín hiệu MACD
    macd_signals_df['timestamp'] = df['timestamp']

    # Chuyển đổi dữ liệu thành dạng dài (long format)
    macd_signals_long = pd.melt(macd_signals_df, id_vars=['timestamp'], var_name='symbol_column', value_name='macd')

    # Tách 'symbol_column' thành 'MACD_Sk_Lk' và 'symbol'
    macd_signals_long['symbol'] = macd_signals_long['symbol_column'].apply(lambda x: x.split('_')[-1])
    macd_signals_long['MACD_Sk_Lk'] = macd_signals_long['symbol_column'].apply(lambda x: '_'.join(x.split('_')[:3]))

    # Xóa cột 'symbol_column' sau khi tách ra các cột riêng
    macd_signals_long = macd_signals_long.drop(columns=['symbol_column'])

    return macd_signals_long

# Hàm tính toán và lưu tín hiệu MACD cho các tập dữ liệu
def calculate_and_save_macd_signals(train, val,test periods = periods, prefix=''):
    datasets = {
    'train': train,
    'val': val,
    'test': test
    }
    # Tạo thư mục 'return' nếu chưa tồn tại
    if not os.path.exists('data/return'):
        os.makedirs('data/return')

    # Duyệt qua các DataFrame và tính toán tín hiệu MACD
    for name, df in dfs.items():
        macd_signals = compute_macd_signals(df, periods['short'], periods['long'])
        macd_signals.to_csv(f'data/return/normalized_macd_{prefix}_{name}.csv', index=False)
        print(f"Đã lưu file data/return/normalized_macd_{prefix}_{name}.csv")

# Các khoảng thời gian ngắn và dài (chỉ tính ba cặp 8-24, 16-48, 32-96)
periods = {
    'short': [8, 16, 32],
    'long': [24, 48, 96]
}

class MACD:
    def __init__(self, prices, params, rebalance_months):
        self.prices = prices
        self.params = params
        self.rebalance_months = rebalance_months

    def macd(self):
        macds = []
        for span_short, span_long in zip(self.params["macd_span_short"], self.params["macd_span_long"]):
            halflife_short = np.log(0.5) / np.log(1 - 1 / span_short)
            halflife_long = np.log(0.5) / np.log(1 - 1 / span_long)

            macd_indicator_short = self.prices.ewm(halflife=halflife_short).mean()
            macd_indicator_long = self.prices.ewm(halflife=halflife_long).mean()

            macd_indicator = macd_indicator_short - macd_indicator_long
            macd_indicator /= self.prices.rolling(window=self.params["macd_std_short"]).std()
            macd_indicator /= self.prices.rolling(window=self.params["macd_std_long"]).std()

            macds.append(macd_indicator)

        macd_combined = sum(macds) / len(macds)
        return macd_combined

    def weights(self, n):
        n_top, n_bottom = n // 2, n // 2
        macd_combined = self.macd()
        macd_combined = macd_combined[macd_combined.index.isin(self.rebalance_months)]

        weights = pd.DataFrame(0, index=macd_combined.index, columns=self.prices.columns)

        for date in macd_combined.index:
            weights_monthly = macd_combined.loc[date].to_numpy()
            largest_indices = np.argpartition(weights_monthly, -n_top)[-n_top:]
            smallest_indices = np.argpartition(weights_monthly, n_bottom)[:n_bottom]

            weights.loc[date, self.prices.columns[largest_indices]] = 1
            weights.loc[date, self.prices.columns[smallest_indices]] = -1

        weights.to_csv(f"data/predictions/csmom_macd_weights_monthly.csv", header=True)

        long_positions = weights[weights == 1].stack().reset_index()
        long_positions.columns = ['date', 'symbol', 'weight']
        long_positions = long_positions[['date', 'symbol']]

        short_positions = weights[weights == -1].stack().reset_index()
        short_positions.columns = ['date', 'symbol', 'weight']
        short_positions = short_positions[['date', 'symbol']]
        if not os.path.exists('data/weights'):
            os.makedirs('data/weights')

        long_positions.to_csv(f"data/weights/long_MACD.csv", index=False)
        short_positions.to_csv(f"data/weights/short_MACD.csv", index=False)
        # Tạo file long_short.csv chứa cả long và short
        long_short_positions = pd.concat([long_positions, short_positions])
        long_short_positions['position'] = long_short_positions.apply(lambda x: 1 if x['symbol'] in long_positions['symbol'].values else -1, axis=1)

        # Lưu file long_short
        long_short_positions.to_csv(f"data/weights/long_short_MACD.csv", index=False)
        return weights



def calculate_macd_weights(train_data, top_n, params =params):
    """
    Tính toán trọng số MACD dựa trên dữ liệu đầu vào, tham số MACD và số lượng tài sản top N.

    Args:
        train_data (pd.DataFrame): Dữ liệu OHLCV với cột timestamp.
        params (dict): Tham số MACD bao gồm:
            - macd_span_short: Danh sách các khoảng thời gian ngắn.
            - macd_span_long: Danh sách các khoảng thời gian dài.
            - macd_std_short: Giá trị chuẩn ngắn hạn.
            - macd_std_long: Giá trị chuẩn dài hạn.
        top_n (int): Số lượng tài sản để lấy trọng số.

    Returns:
        pd.DataFrame: DataFrame chứa timestamp, symbols, và weights.
    """
    # Đảm bảo timestamp được xử lý đúng định dạng
    train_data["timestamp"] = pd.to_datetime(train_data["timestamp"])
    train_data = train_data.set_index("timestamp")

    # Xác định ngày tái cân bằng
    start_date = train_data.index.min().strftime('%Y-%m-%d')
    end_date = train_data.index.max().strftime('%Y-%m-%d')
    rebalance_days = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Khởi tạo MACD và tính toán trọng số (giả định MACD là một lớp đã được định nghĩa)
    macd = MACD(train_data, params, rebalance_days)
    weights_df = macd.weights(top_n)

    # Tạo DataFrame mới chứa các cột timestamp, symbols và weights
    result = []
    for idx, row in weights_df.iterrows():
        timestamp = row.name  # Timestamp từ chỉ số hàng
        symbols = [col for col in weights_df.columns if row[col] != 0]  # Lọc các symbol có trọng số khác 0
        weights = [row[symbol] for symbol in symbols]  # Lấy danh sách trọng số tương ứng

        # Thêm dòng dữ liệu vào danh sách kết quả
        result.append([timestamp, symbols, weights])

    # Tạo DataFrame kết quả
    result_df = pd.DataFrame(result, columns=["timestamp", "symbols", "weights"])

    return result_df


class CSMOM:
    def __init__(self, returns, rebalance_days):
        self.returns = returns
        self.rebalance_days = rebalance_days

    def weights(self, lookback, n, data_type):
        n_top, n_bottom = int(n / 2), int(n / 2)

        if lookback > 1:
            cumulative_returns = self.returns.rolling(lookback).apply(lambda x: (1 + x).prod() - 1)
        else:
            cumulative_returns = self.returns.copy()

        cumulative_returns = cumulative_returns[cumulative_returns.index.isin(self.rebalance_days)]

        long_weights = pd.DataFrame(0, index=cumulative_returns.index, columns=self.returns.columns)
        short_weights = pd.DataFrame(0, index=cumulative_returns.index, columns=self.returns.columns)

        for date in cumulative_returns.index:
            weights_daily = cumulative_returns.loc[date].to_numpy()

            largest_indices = np.argpartition(weights_daily, -n_top)[-n_top:]
            smallest_indices = np.argpartition(weights_daily, n_bottom)[:n_bottom:]

            long_weights.loc[date, self.returns.columns[largest_indices]] = 1
            short_weights.loc[date, self.returns.columns[smallest_indices]] = -1

        signals = []

        for date in cumulative_returns.index:
            long_today = long_weights.loc[date][long_weights.loc[date] == 1].index.tolist()
            short_today = short_weights.loc[date][short_weights.loc[date] == -1].index.tolist()

            combined_symbols = long_today + short_today
            weights_today = [1] * len(long_today) + [-1] * len(short_today)

            signals.append([date, combined_symbols, weights_today])

        signals_df = pd.DataFrame(signals, columns=["date", "symbols", "weights"])

        # Tạo tên file CSV theo loại tập dữ liệu
        file_name = f"data/predictions/classic_momentum_{data_type}_{int(lookback / 21)}m.csv"
        signals_df.to_csv(file_name, index=False)

        return signals_df

import os
import pandas as pd
from datetime import datetime

def save_dataframe_to_csv(df, output_file):
    """
    Lưu DataFrame vào file CSV, tạo thư mục nếu chưa tồn tại.

    Args:
        df (pd.DataFrame): DataFrame cần lưu.
        output_file (str): Đường dẫn file CSV để lưu.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Kết quả đã được lưu tại: {output_file}")


def calculate_macd_weights(data, top_n, output_file=None,params =params):
    """
    Tính toán trọng số MACD dựa trên dữ liệu đầu vào, tham số MACD và số lượng tài sản top N.

    Args:
        data (pd.DataFrame): Dữ liệu OHLCV với cột timestamp.
        params (dict): Tham số MACD bao gồm:
            - macd_span_short: Danh sách các khoảng thời gian ngắn.
            - macd_span_long: Danh sách các khoảng thời gian dài.
            - macd_std_short: Giá trị chuẩn ngắn hạn.
            - macd_std_long: Giá trị chuẩn dài hạn.
        top_n (int): Số lượng tài sản để lấy trọng số.
        output_file (str, optional): Đường dẫn file để lưu kết quả CSV. Nếu không, chỉ trả về DataFrame.

    Returns:
        pd.DataFrame: DataFrame chứa timestamp, symbols, và weights.
    """
    # Đảm bảo timestamp được xử lý đúng định dạng
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.set_index("timestamp")

    # Xác định ngày tái cân bằng dựa trên khoảng thời gian của dữ liệu
    rebalance_days = pd.date_range(start=data.index.min(), end=data.index.max(), freq='MS')

    # Khởi tạo MACD và tính toán trọng số
    macd = MACD(data, params, rebalance_days)
    weights_df = macd.weights(top_n)

    # Tạo DataFrame chứa timestamp, symbols, và weights
    result = [
        {"timestamp": row.name, "symbols": [col for col in weights_df.columns if row[col] != 0],
         "weights": [row[col] for col in weights_df.columns if row[col] != 0]}
        for _, row in weights_df.iterrows()
    ]
    result_df = pd.DataFrame(result)

    # Lưu file nếu có đường dẫn
    if output_file:
        save_dataframe_to_csv(result_df, output_file)

    return result_df

lookbacks = [21]
def calculate_csmom_weights(data, top_n, output_dir=None, data_type=None, lookbacks=lookbacks):
    """
    Tính toán trọng số CSMOM dựa trên dữ liệu đầu vào và các khoảng lookback.

    Args:
        data (pd.DataFrame): Dữ liệu OHLCV với cột timestamp.
        lookbacks (list): Danh sách các khoảng lookback tính bằng ngày.
        top_n (int): Số lượng tài sản để lấy trọng số.
        output_dir (str, optional): Thư mục để lưu kết quả. Nếu không, chỉ trả về dictionary.
        data_type (str, optional): Loại dữ liệu (train, val, test) để tạo tên cho file.

    Returns:
        dict: Dictionary chứa DataFrame trọng số cho từng khoảng lookback.
    """
    # Đảm bảo timestamp được xử lý đúng định dạng
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.set_index("timestamp")

    # Xác định ngày tái cân bằng dựa trên khoảng thời gian của dữ liệu
    rebalance_days = pd.date_range(start=data.index.min(), end=data.index.max(), freq='MS')

    # Tính toán lợi nhuận
    returns = data.pct_change()

    # Khởi tạo CSMOM và tính toán trọng số
    csmom = CSMOM(returns, rebalance_days)
    weights_dict = {}

    for lookback in lookbacks:
        signals_df = csmom.weights(lookback, top_n, data_type)

        # Lưu DataFrame vào weights_dict
        weights_dict[lookback] = signals_df

        # Lưu kết quả vào thư mục nếu có output_dir
        if output_dir:
            output_file = os.path.join(output_dir, f"{data_type}_csmom_weights_{lookback}_days.csv")
            save_dataframe_to_csv(signals_df, output_file)

    return weights_dict








