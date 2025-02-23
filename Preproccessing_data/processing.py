import os
import pandas as pd

def merge_data(data_dir = './data_2022_2024'):
    # Danh sách để lưu DataFrames
    dataframes = []

    # Duyệt qua tất cả các tệp CSV trong thư mục
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)
            df['symbol'] = filename.split('_')[0]  # Thêm cột 'symbol' để nhận diện
            dataframes.append(df)

    # Gộp tất cả DataFrames thành một DataFrame duy nhất
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Lưu DataFrame đã gộp thành một tệp CSV
    merged_filename = 'merged_ohlcv_data.csv'
    merged_df.to_csv(merged_filename, index=False)
    print("merging data is finish")
    print('\n')
    
def processing_data(df =pd.read_csv('data_clean.csv')):
    # Chuyển đổi cột 'Time' về kiểu datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    # 1. Kiểm tra thông tin cơ bản của DataFrame
    print("Thông tin cơ bản của DataFrame:")
    print(df.info())

    # 2. Kiểm tra các giá trị bị thiếu
    print("\nGiá trị bị thiếu trong các cột:")
    print(df.isnull().sum())
    
    print("\nDữ liệu trùng lặp:")
    print(df.duplicated().sum())  
    # Loại bỏ các hàng trùng lặp
    df_no_duplicates = df.drop_duplicates()
    print("\nDataFrame sau khi loại bỏ trùng lặp:")
    print(df_no_duplicates.info())
    
    # 4. Kiểm tra dữ liệu ngoại lai (outliers) bằng phương pháp IQR
    # Chọn cột volume để kiểm tra ngoại lai
    volume_col = 'volume'

    # Tính toán Q1, Q3 và IQR cho cột volume
    Q1_volume = df[volume_col].quantile(0.25)
    Q3_volume = df[volume_col].quantile(0.75)
    IQR_volume = Q3_volume - Q1_volume

    # Xác định các ngoại lai trong cột volume
    outliers_volume = (df[volume_col] < (Q1_volume - 1.5 * IQR_volume)) | (df[volume_col] > (Q3_volume + 1.5 * IQR_volume))

    # In ra số lượng ngoại lai trong cột volume
    print("\nSố lượng ngoại lai trong cột volume:")
    print(outliers_volume.sum())

    # Loại bỏ các hàng có ngoại lai trong cột volume
    df_no_outliers_volume = df[~outliers_volume]
    print("\nDataFrame sau khi loại bỏ ngoại lai trong cột volume:")
    print(df_no_outliers_volume.info())
    


def clean_data():
    data = pd.read_csv("merged_ohlcv_data.csv")
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    duplicates = data[data.duplicated(subset=['timestamp', 'symbol'], keep=False)]
    if not duplicates.empty:
        print("Các bản ghi trùng lặp:")
        print(duplicates)

    data = data.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')

    pivoted_data = data.pivot(index='timestamp', columns='symbol', values='close')

    pivoted_data.reset_index(inplace=True)

    pivoted_data.to_csv("pivoted_data.csv", index=False)