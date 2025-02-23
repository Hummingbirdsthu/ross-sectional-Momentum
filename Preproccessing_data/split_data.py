from pybit.unified_trading import HTTP
import pandas as pd
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def split_data_merged(df_original = pd.read_csv('./merged_ohlcv_data.csv')):
    
    # Chuyển cột 'timestamp' sang kiểu datetime
    df_original['timestamp'] = pd.to_datetime(df_original['timestamp'])
    # Lọc các tập dữ liệu theo thời gian
    train_or = df_original[(df_original['timestamp'] >= '2022-01-01') & (df_original['timestamp'] < '2023-01-01')]
    val_or = df_original[(df_original['timestamp'] >= '2023-01-01') & (df_original['timestamp'] < '2023-06-01')]
    test_or = df_original[(df_original['timestamp'] >= '2023-07-01') & (df_original['timestamp'] < '2024-09-01')]
    return train_or , val_or,test_or 