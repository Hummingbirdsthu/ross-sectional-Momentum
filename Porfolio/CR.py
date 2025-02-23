# Tạo các DataFrame cho Cumulative_long_short, Cumulative_long, và Cumulative_short
import pandas as pd 
import matplotlib.pyplot as plt
def plot_CR_Return():
    Returns =pd.read_csv('profit_CR/Return.csv')
    Cumulative =Returns.copy()

    # Tính lợi nhuận tích lũy cho từng nhóm
    # Cumulative_long_short: Dữ liệu gốc
    Cumulative['profit']= (Cumulative['weight'] *  Cumulative['raw_return'])

    Cumulative['profit']= (Cumulative['raw_return'])

    Cumulative_long_short = Cumulative.copy() 
    Cumulative_long_short['symbol_count'] = Cumulative_long_short.groupby('Month')['symbol'].transform('nunique')
    Cumulative_long_short = Cumulative_long_short.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_long_short['profit_t'] = Cumulative_long_short['profit'] / Cumulative_long_short['symbol_count']
    Cumulative_long_short['Cumulative'] = (1 + Cumulative_long_short['profit_t']).cumprod() - 1

    # Cumulative_long: Dữ liệu với weight == 1
    Cumulative_long = Cumulative[Cumulative['weight'] == 1].copy() 
    Cumulative_long['symbol_count'] = Cumulative_long.groupby('Month')['symbol'].transform('nunique')
    Cumulative_long = Cumulative_long.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_long['profit_t'] = Cumulative_long['profit'] / Cumulative_long['symbol_count']
    Cumulative_long['Cumulative'] = (1 + Cumulative_long['profit_t']).cumprod() - 1

    # Cumulative_short: Dữ liệu với weight == -1
    Cumulative_short = Cumulative[Cumulative['weight'] == -1].copy() 
    Cumulative_short['symbol_count'] = Cumulative_short.groupby('Month')['symbol'].transform('nunique')
    Cumulative_short = Cumulative_short.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_short['profit_t'] = Cumulative_short['profit'] / Cumulative_short['symbol_count']
    Cumulative_short['Cumulative'] = (1 + Cumulative_short['profit_t']).cumprod() - 1


    #Cumulative_equal_weight
    Cumulative_equal_weight = pd.read_csv('profit_CR/equal_weights_return.csv')



    # Đảm bảo cột 'Month' có định dạng datetime 
    Cumulative_long_short['Month'] = pd.to_datetime(Cumulative_long_short['Month'])
    Cumulative_long['Month'] = pd.to_datetime(Cumulative_long['Month'])
    Cumulative_short['Month'] = pd.to_datetime(Cumulative_short['Month'])
    Cumulative_equal_weight['Month']= pd.to_datetime(Cumulative_equal_weight['Month'])

    # Vẽ biểu đồ
    plt.figure(figsize=(14, 8))  # Tăng kích thước biểu đồ

    # Vẽ Cumulative_long_short
    plt.plot(Cumulative_long_short['Month'], Cumulative_long_short['Cumulative'], marker='s', linestyle='-', color='#8B4513', linewidth=2, label='Cumulative (Long/Short)')

    # Vẽ Cumulative_long
    plt.plot(Cumulative_long['Month'], Cumulative_long['Cumulative'], marker='o', linestyle='-', color='#1E90FF', linewidth=2, label='Cumulative (Long)')

    # Vẽ Cumulative_short
    plt.plot(Cumulative_short['Month'], Cumulative_short['Cumulative'], marker='^', linestyle='-', color='#FF6347', linewidth=2, label='Cumulative (Short)')

    # Vẽ Equal weight với thay đổi nét và màu
    plt.plot(Cumulative_equal_weight['Month'], Cumulative_equal_weight['Cumulative'], marker='^', linestyle='--', color='#32CD32', linewidth=3, label='Cumulative (Equal weight)')

    # Thêm tiêu đề và nhãn với kích thước font lớn hơn
    plt.title('Cumulative Return by Return(Long, Short, Long/Short and Equal Weight)', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)

    # Tùy chỉnh trục x và hiển thị nhãn trục x là tháng với khoảng cách hợp lý
    plt.xticks(pd.date_range(start=Cumulative['Month'].min(), end=Cumulative['Month'].max(), freq='MS'), rotation=45, fontsize=12)

    # Tăng cỡ chữ cho nhãn trục y và hiển thị các số liệu dễ đọc hơn
    plt.yticks(fontsize=12)

    # Hiển thị lưới và legend với font chữ dễ đọc
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Điều chỉnh khoảng cách giữa các điểm trên đồ thị để rõ ràng hơn
    plt.tight_layout()

    # Lưu và hiển thị biểu đồ
    plt.savefig(f'Profit_CR/CR_Return.png', dpi=300, bbox_inches='tight')

def plot_CR_MACD():
    # Tạo các DataFrame cho Cumulative_long_short, Cumulative_long, và Cumulative_short
    Macd =pd.read_csv('profit_CR/MACD.csv')
    Cumulative =Macd.copy()
    # Tính lợi nhuận tích lũy cho từng nhóm
    # Cumulative_long_short: Dữ liệu gốc
    # Cumulative['profit']= (Cumulative['weight'] *  Cumulative['raw_return'])
    Cumulative['profit']=  Cumulative['raw_return']

    Cumulative_long_short = Cumulative.copy() 
    Cumulative_long_short['symbol_count'] = Cumulative_long_short.groupby('Month')['symbol'].transform('nunique')
    Cumulative_long_short = Cumulative_long_short.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_long_short['profit_t'] = Cumulative_long_short['profit'] / Cumulative_long_short['symbol_count']
    Cumulative_long_short['Cumulative'] = (1 + Cumulative_long_short['profit_t']).cumprod() - 1

    # Cumulative_long: Dữ liệu với weight == 1
    Cumulative_long = Cumulative[Cumulative['weight'] == 1].copy() 
    Cumulative_long['symbol_count'] = Cumulative_long.groupby('Month')['symbol'].transform('nunique')
    Cumulative_long = Cumulative_long.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_long['profit_t'] = Cumulative_long['profit'] / Cumulative_long['symbol_count']
    Cumulative_long['Cumulative'] = (1 + Cumulative_long['profit_t']).cumprod() - 1

    # Cumulative_short: Dữ liệu với weight == -1
    Cumulative_short = Cumulative[Cumulative['weight'] == -1].copy() 
    Cumulative_short['symbol_count'] = Cumulative_short.groupby('Month')['symbol'].transform('nunique')
    Cumulative_short = Cumulative_short.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_short['profit_t'] = Cumulative_short['profit'] / Cumulative_short['symbol_count']
    Cumulative_short['Cumulative'] = (1 + Cumulative_short['profit_t']).cumprod() - 1
    #Cumulative_equal_weight
    Cumulative_equal_weight = pd.read_csv('profit_CR/equal_weight_MACD.csv')

    # Đảm bảo cột 'Month' có định dạng datetime
    Cumulative_long_short['Month'] = pd.to_datetime(Cumulative_long_short['Month'])
    Cumulative_long['Month'] = pd.to_datetime(Cumulative_long['Month'])
    Cumulative_short['Month'] = pd.to_datetime(Cumulative_short['Month'])
    Cumulative_equal_weight['Month'] = pd.to_datetime(Cumulative_equal_weight['Month'])

    # Vẽ biểu đồ
    plt.figure(figsize=(14, 8))  # Tăng kích thước biểu đồ

    # Vẽ Cumulative_long_short
    plt.plot(Cumulative_long_short['Month'], Cumulative_long_short['Cumulative'], marker='s', linestyle='-', color='#8B4513', linewidth=2, label='Cumulative (Long/Short)')

    # Vẽ Cumulative_long
    plt.plot(Cumulative_long['Month'], Cumulative_long['Cumulative'], marker='o', linestyle='-', color='#1E90FF', linewidth=2, label='Cumulative (Long)')

    # Vẽ Cumulative_short
    plt.plot(Cumulative_short['Month'], Cumulative_short['Cumulative'], marker='^', linestyle='-', color='#FF6347', linewidth=2, label='Cumulative (Short)')

    # Vẽ Equal weight với thay đổi nét và màu
    plt.plot(Cumulative_equal_weight['Month'], Cumulative_equal_weight['Cumulative'], marker='^', linestyle='--', color='#32CD32', linewidth=3, label='Cumulative (Equal weight)')
    # Thêm tiêu đề và nhãn với kích thước font lớn hơn
    plt.title('Cumulative Return by MACD(Long, Short, and Long/Short)', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)

    # Tùy chỉnh trục x và hiển thị nhãn trục x là tháng với khoảng cách hợp lý
    plt.xticks(pd.date_range(start=Cumulative['Month'].min(), end=Cumulative['Month'].max(), freq='MS'), rotation=45, fontsize=12)

    # Tăng cỡ chữ cho nhãn trục y và hiển thị các số liệu dễ đọc hơn
    plt.yticks(fontsize=12)

    # Hiển thị lưới và legend với font chữ dễ đọc
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Điều chỉnh khoảng cách giữa các điểm trên đồ thị để rõ ràng hơn
    plt.tight_layout()

    # Lưu và hiển thị biểu đồ
    plt.savefig(f'Profit_CR/CR_MACD.png', dpi=300, bbox_inches='tight')

def plot_CR_LambdaMART():
    lambdaa =pd.read_csv('profit_CR/lambda.csv')
    Cumulative =lambdaa.copy()
    Cumulative_long_short =Cumulative.copy()

    Cumulative_long_short= Cumulative_long_short[Cumulative_long_short['predict_rank'] != 0 ]
    Cumulative_long_short['Month'] = pd.to_datetime(Cumulative_long_short['Month'])
    # Cumulative_long_short['profit']= Cumulative_long_short['predict_rank'] *Cumulative_long_short['raw_return']
    Cumulative_long_short['profit']= Cumulative_long_short['raw_return']
    Cumulative_long_short['symbol_count'] = Cumulative_long_short.groupby('Month')['symbol'].transform('nunique')

    Cumulative_long_short = Cumulative_long_short.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_long_short['profit_t'] = Cumulative_long_short['profit'] / Cumulative_long_short['symbol_count']
    Cumulative_long_short['Cumulative'] = (1 + Cumulative_long_short['profit_t']).cumprod() - 1

    ### long  

    Cumulative_long =Cumulative[Cumulative['predict_rank'] == 1 ].copy()


    Cumulative_long['Month'] = pd.to_datetime(Cumulative_long['Month'])
    # Cumulative_long['profit']= Cumulative_long['predict_rank'] *Cumulative_long['raw_return']
    Cumulative_long['profit']= Cumulative_long['raw_return']
    Cumulative_long['symbol_count'] = Cumulative_long.groupby('Month')['symbol'].transform('nunique')

    Cumulative_long = Cumulative_long.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_long['profit_t'] = Cumulative_long['profit'] / Cumulative_long['symbol_count']
    Cumulative_long['Cumulative'] = (1 + Cumulative_long['profit_t']).cumprod() - 1
    ## short



    Cumulative_short =Cumulative[Cumulative['predict_rank'] == -1 ].copy()


    Cumulative_short['Month'] = pd.to_datetime(Cumulative_short['Month'])
    # Cumulative_short['profit']= Cumulative_short['predict_rank'] *Cumulative_short['raw_return']
    Cumulative_short['profit']= Cumulative_short['raw_return']
    Cumulative_short['symbol_count'] = Cumulative_short.groupby('Month')['symbol'].transform('nunique')

    Cumulative_short = Cumulative_short.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_short['profit_t'] = Cumulative_short['profit'] / Cumulative_short['symbol_count']
    Cumulative_short['Cumulative'] = (1 + Cumulative_short['profit_t']).cumprod() - 1
    ## 
    Cumulative_equal_weight = pd.read_csv('profit_CR/EW_LambdaMART.csv')
    Cumulative_equal_weight['Month'] = pd.to_datetime(Cumulative_equal_weight['Month'])
    # Vẽ biểu đồ
    plt.figure(figsize=(14, 8))  # Tăng kích thước biểu đồ


    Cumulative_long_short["Cumulative"].iloc[0] = 0
    Cumulative_long["Cumulative"].iloc[0] = 0
    Cumulative_short["Cumulative"].iloc[0] = 0
    Cumulative_equal_weight["Cumulative"].iloc[0] = 0


    # Vẽ Cumulative_long_short
    plt.plot(Cumulative_long_short['Month'], Cumulative_long_short['Cumulative'], marker='s', linestyle='-', color='#8B4513', linewidth=2, label='Cumulative (Long/Short)')

    # Vẽ Cumulative_long
    plt.plot(Cumulative_long['Month'], Cumulative_long['Cumulative'], marker='o', linestyle='-', color='#1E90FF', linewidth=2, label='Cumulative (Long)')

    # Vẽ Cumulative_short
    plt.plot(Cumulative_short['Month'], Cumulative_short['Cumulative'], marker='^', linestyle='-', color='#FF6347', linewidth=2, label='Cumulative (Short)')

    # Vẽ Equal weight với thay đổi nét và màu
    plt.plot(Cumulative_equal_weight['Month'], Cumulative_equal_weight['Cumulative'], marker='^', linestyle='--', color='#32CD32', linewidth=3, label='Cumulative (Equal weight)')

    # Thêm tiêu đề và nhãn với kích thước font lớn hơn
    plt.title('Cumulative Return by Lambda(Long, Short, Long/Short and Equal Weight)', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)

    # Tùy chỉnh trục x và hiển thị nhãn trục x là tháng với khoảng cách hợp lý
    plt.xticks(pd.date_range(start=Cumulative['Month'].min(), end=Cumulative['Month'].max(), freq='MS'), rotation=45, fontsize=12)

    # Tăng cỡ chữ cho nhãn trục y và hiển thị các số liệu dễ đọc hơn
    plt.yticks(fontsize=12)

    # Hiển thị lưới và legend với font chữ dễ đọc
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Điều chỉnh khoảng cách giữa các điểm trên đồ thị để rõ ràng hơn
    plt.tight_layout()

    # Lưu và hiển thị biểu đồ
    plt.savefig(f'Profit_CR/CR_Lambda.png', dpi=300, bbox_inches='tight')
def plot_CR_RandomForest():
    # Giả sử bạn đã có DataFrame Cumulative với cột 'Month' và 'Cumulative'
    rand_rank = pd.read_csv('RandomForest_rank.csv')
    data2 = rand_rank[rand_rank['Month']>= '2023-06']
    # data2['Month'] = data2['Month'].dt.to_period('M')
    Cumulative =data2.copy()

    Cumulative_long_short =Cumulative.copy()

    Cumulative_long_short= Cumulative_long_short[Cumulative_long_short['PredictedRank'] != 0 ]
    Cumulative_long_short['Month'] = pd.to_datetime(Cumulative_long_short['Month'])
    # Cumulative_long_short['profit']= Cumulative_long_short['PredictedRank'] *Cumulative_long_short['Returns']
    Cumulative_long_short['profit']= Cumulative_long_short['Returns']
    Cumulative_long_short['symbol_count'] = Cumulative_long_short.groupby('Month')['Symbol'].transform('nunique')

    Cumulative_long_short = Cumulative_long_short.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_long_short['profit_t'] = Cumulative_long_short['profit'] / Cumulative_long_short['symbol_count']
    Cumulative_long_short['Cumulative'] = (1 + Cumulative_long_short['profit_t']).cumprod() - 1

    ### long  

    Cumulative_long =Cumulative[Cumulative['PredictedRank'] == 1 ].copy()


    Cumulative_long['Month'] = pd.to_datetime(Cumulative_long['Month'])
    # Cumulative_long['profit']= Cumulative_long['PredictedRank'] *Cumulative_long['Returns']

    Cumulative_long['profit']= Cumulative_long['Returns']
    Cumulative_long['symbol_count'] = Cumulative_long.groupby('Month')['Symbol'].transform('nunique')

    Cumulative_long = Cumulative_long.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_long['profit_t'] = Cumulative_long['profit'] / Cumulative_long['symbol_count']
    Cumulative_long['Cumulative'] = (1 + Cumulative_long['profit_t']).cumprod() - 1
    ## short



    Cumulative_short =Cumulative[Cumulative['PredictedRank'] == -1 ].copy()


    Cumulative_short['Month'] = pd.to_datetime(Cumulative_short['Month'])
    # Cumulative_short['profit']= Cumulative_short['PredictedRank'] *Cumulative_short['Returns']
    Cumulative_short['profit']= Cumulative_short['Returns']
    Cumulative_short['symbol_count'] = Cumulative_short.groupby('Month')['Symbol'].transform('nunique')

    Cumulative_short = Cumulative_short.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_short['profit_t'] = Cumulative_short['profit'] / Cumulative_short['symbol_count']
    Cumulative_short['Cumulative'] = (1 + Cumulative_short['profit_t']).cumprod() - 1


    ### equal_weight
    Cumulative_equal_weight = pd.read_csv('profit_CR/EW_RF.csv')
    Cumulative_equal_weight['Month'] = pd.to_datetime(Cumulative_equal_weight['Month'])



    Cumulative_long_short["Cumulative"].iloc[0] = 0
    Cumulative_long["Cumulative"].iloc[0] = 0
    Cumulative_short["Cumulative"].iloc[0] = 0
    Cumulative_equal_weight["Cumulative"].iloc[0] = 0

    # Vẽ biểu đồ
    plt.figure(figsize=(14, 8))  # Tăng kích thước biểu đồ

    # Vẽ Cumulative_long_short
    plt.plot(Cumulative_long_short['Month'], Cumulative_long_short['Cumulative'], marker='s', linestyle='-', color='#8B4513', linewidth=2, label='Cumulative (Long/Short)')

    # Vẽ Cumulative_long
    plt.plot(Cumulative_long['Month'], Cumulative_long['Cumulative'], marker='o', linestyle='-', color='#1E90FF', linewidth=2, label='Cumulative (Long)')

    # Vẽ Cumulative_short
    plt.plot(Cumulative_short['Month'], Cumulative_short['Cumulative'], marker='^', linestyle='-', color='#FF6347', linewidth=2, label='Cumulative (Short)')

    # Vẽ Equal weight với thay đổi nét và màu
    plt.plot(Cumulative_equal_weight['Month'], Cumulative_equal_weight['Cumulative'], marker='^', linestyle='--', color='#32CD32', linewidth=3, label='Cumulative (Equal weight)')

    # Thêm tiêu đề và nhãn với kích thước font lớn hơn
    plt.title('Cumulative Return by Random Forest(Long, Short, Long/Short and Equal Weight)', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)

    # Tùy chỉnh trục x và hiển thị nhãn trục x là tháng với khoảng cách hợp lý
    plt.xticks(pd.date_range(start=Cumulative['Month'].min(), end=Cumulative['Month'].max(), freq='MS'), rotation=45, fontsize=12)

    # Tăng cỡ chữ cho nhãn trục y và hiển thị các số liệu dễ đọc hơn
    plt.yticks(fontsize=12)

    # Hiển thị lưới và legend với font chữ dễ đọc
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Điều chỉnh khoảng cách giữa các điểm trên đồ thị để rõ ràng hơn
    plt.tight_layout()

    # Lưu và hiển thị biểu đồ
    plt.savefig(f'Profit_CR/CR_RF.png', dpi=300, bbox_inches='tight')

def plot_CR_ListNet():
    # Giả sử bạn đã có DataFrame Cumulative với cột 'Month' và 'Cumulative'
    ListNet= pd.read_csv('profit_CR/ListNet_rank.csv')
    data2 = ListNet[plot_by_ListNet['Month']>= '2023-06']
    # data2['Month'] = data2['Month'].dt.to_period('M')
    Cumulative =data2.copy()
    Cumulative_long_short =Cumulative.copy()
    Cumulative_long_short= Cumulative_long_short[Cumulative_long_short['PredictedRank'] != 0 ]
    Cumulative_long_short['Month'] = pd.to_datetime(Cumulative_long_short['Month'])
    # Cumulative_long_short['profit']= Cumulative_long_short['PredictedRank'] *Cumulative_long_short['Returns']
    Cumulative_long_short['profit']= Cumulative_long_short['Returns']
    Cumulative_long_short['symbol_count'] = Cumulative_long_short.groupby('Month')['Symbol'].transform('nunique')

    Cumulative_long_short = Cumulative_long_short.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_long_short['profit_t'] = Cumulative_long_short['profit'] / Cumulative_long_short['symbol_count']
    Cumulative_long_short['Cumulative'] = (1 + Cumulative_long_short['profit_t']).cumprod() - 1

    ### long  

    Cumulative_long =Cumulative[Cumulative['PredictedRank'] == 1 ].copy()


    Cumulative_long['Month'] = pd.to_datetime(Cumulative_long['Month'])
    # Cumulative_long['profit']= Cumulative_long['PredictedRank'] *Cumulative_long['Returns']

    Cumulative_long['profit']= Cumulative_long['Returns']
    Cumulative_long['symbol_count'] = Cumulative_long.groupby('Month')['Symbol'].transform('nunique')

    Cumulative_long = Cumulative_long.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_long['profit_t'] = Cumulative_long['profit'] / Cumulative_long['symbol_count']
    Cumulative_long['Cumulative'] = (1 + Cumulative_long['profit_t']).cumprod() - 1
    ## short



    Cumulative_short =Cumulative[Cumulative['PredictedRank'] == -1 ].copy()


    Cumulative_short['Month'] = pd.to_datetime(Cumulative_short['Month'])
    # Cumulative_short['profit']= Cumulative_short['PredictedRank'] *Cumulative_short['Returns']
    Cumulative_short['profit']= Cumulative_short['Returns']
    Cumulative_short['symbol_count'] = Cumulative_short.groupby('Month')['Symbol'].transform('nunique')

    Cumulative_short = Cumulative_short.groupby('Month').agg({
        'profit': 'sum',
        'symbol_count': 'first'  # Lấy số lượng symbol (đã tính ở trên)
    }).reset_index()
    Cumulative_short['profit_t'] = Cumulative_short['profit'] / Cumulative_short['symbol_count']
    Cumulative_short['Cumulative'] = (1 + Cumulative_short['profit_t']).cumprod() - 1


    ### equal_weight
    Cumulative_equal_weight = pd.read_csv('profit_CR/EW_RF.csv')
    Cumulative_equal_weight['Month'] = pd.to_datetime(Cumulative_equal_weight['Month'])


    Cumulative_long_short["Cumulative"].iloc[0] = 0
    Cumulative_long["Cumulative"].iloc[0] = 0
    Cumulative_short["Cumulative"].iloc[0] = 0
    Cumulative_equal_weight["Cumulative"].iloc[0] = 0

    # Vẽ biểu đồ
    plt.figure(figsize=(14, 8))  # Tăng kích thước biểu đồ

    # Vẽ Cumulative_long_short
    plt.plot(Cumulative_long_short['Month'], Cumulative_long_short['Cumulative'], marker='s', linestyle='-', color='#8B4513', linewidth=2, label='Cumulative (Long/Short)')

    # Vẽ Cumulative_long
    plt.plot(Cumulative_long['Month'], Cumulative_long['Cumulative'], marker='o', linestyle='-', color='#1E90FF', linewidth=2, label='Cumulative (Long)')

    # Vẽ Cumulative_short
    plt.plot(Cumulative_short['Month'], Cumulative_short['Cumulative'], marker='^', linestyle='-', color='#FF6347', linewidth=2, label='Cumulative (Short)')

    # Vẽ Equal weight với thay đổi nét và màu
    plt.plot(Cumulative_equal_weight['Month'], Cumulative_equal_weight['Cumulative'], marker='^', linestyle='--', color='#32CD32', linewidth=3, label='Cumulative (Equal weight)')

    # Thêm tiêu đề và nhãn với kích thước font lớn hơn
    plt.title('Cumulative Return by ListNet(Long, Short, Long/Short and Equal Weight)', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)

    # Tùy chỉnh trục x và hiển thị nhãn trục x là tháng với khoảng cách hợp lý
    plt.xticks(pd.date_range(start=Cumulative['Month'].min(), end=Cumulative['Month'].max(), freq='MS'), rotation=45, fontsize=12)

    # Tăng cỡ chữ cho nhãn trục y và hiển thị các số liệu dễ đọc hơn
    plt.yticks(fontsize=12)

    # Hiển thị lưới và legend với font chữ dễ đọc
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Điều chỉnh khoảng cách giữa các điểm trên đồ thị để rõ ràng hơn
    plt.tight_layout()

    # Lưu và hiển thị biểu đồ
    plt.savefig(f'Profit_CR/CR_ListNet.png', dpi=300, bbox_inches='tight')