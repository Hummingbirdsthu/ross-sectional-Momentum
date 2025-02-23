import asyncio
from get_data.get_data import fetch_ohlcv_data
from Preproccessing_data.processing import merge_data
from Preproccessing_data.split_data import split_data_clearn, split_data_merged
from return_macd.return_macd import return_ , macd_
from long_short_macd_orgi.macd_mom_orgi.macd import weight_macd
from long_short_macd_orgi.macd_mom_orgi.mom_orgi import weight_mom_orgi
from long_short_macd_orgi.data_feature import long,long_short,long_short1, only_long,only_short,short
from statistical.statistical import creat_file_anova_results , plot_3_6_month_return,performance_metrics
from cumulative_return.cumu_return import plot_cumulative_return

def backtesting():
    # load data
    #asyncio.run(fetch_ohlcv_data())
    # merge_data 
    merge_data()
    # #split data clearn
    train ,val ,test =split_data_clearn()
    # #split data merge
    train_or,val_or,test_or  = split_data_merged()
    # # Lợi nhuận lợi suất tích lũy trong khung thời gian 3,6,9,12 tháng 
    
    # print('----------------')
    return_(train.copy())
    # # chỉ số macd
    # print('----------------')
    macd_(train.copy())
    # print('done')
    # #trọng số macd
    weight_macd(train.copy())
    # #trọng số orgi
    # print('----------------')
    weight_mom_orgi(train.copy())
    long.long(train_or.copy())
    # #short
    # print('----------------')
    short.short(train_or.copy())
    # #long short 
    # print('----------------')
    long_short.long_short_macd(train_or.copy())
    long_short.long_short_return(train_or.copy())
    # print('----------------')
   
    
    # print('----------------')
    # creat_file_anova_results()# tạo các chỉ số thống kê 
    plot_3_6_month_return()# plot 
    # performance_metrics()
    plot_cumulative_return()
    # Tạo các thanh trượt (sliders) cho trục X và Y

if __name__ == "__main__":
    backtesting()