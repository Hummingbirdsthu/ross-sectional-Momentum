import pandas as pd
import streamlit as st
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
# Đọc file CSS và thêm vào ứng dụng
# with open("styles.css", encoding="utf-8") as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Thêm tiêu đề cho ứng dụng với CSS đã định dạng
st.markdown('<h1 class="title">ỨNG DỤNG KIỂM TRA HIỆU QUẢ CÁC PHƯƠNG PHÁP VÀ CHIẾN LƯỢC </h1>', unsafe_allow_html=True)
st.markdown("""
**Vấn đề nghiên cứu** là liệu các chiến lược động lượng có thể duy trì hiệu
quả trong bối cảnh thị trường bị ảnh hưởng bởi các sự kiện lớn hay
không? Cụ thể hơn, CSM có còn là một chiến lược hợp lý khi các loại tài sản
có thể bị ảnh hưởng bởi những thay đổi đột ngột trong điều kiện thị trường?
Nghiên cứu này sẽ tập trung phân tích hiệu quả của các chiến lược động lượng,
so sánh chúng nhằm cung cấp một cái nhìn toàn diện hơn về cách thông qua
nghiên cứu chỉ so sánh bốn phương thức: **chỉ long**, **chỉ short**, **long short**, và **trọng
số bằng nhau**, kết hợp mô hình xếp hạng để tăng hiệu suất.
""")
# Tạo thanh công cụ nằm ngang với các tùy chọn mà không có chữ "Chọn phần"
tab_selection = st.radio("", ["Assets", "Phương pháp và chiến lược"], horizontal=True)
if tab_selection == "Assets":
    
    st.markdown(""" Dữ liệu được thu thập bao gồm các cặp giao dịch tiền mã hóa trên nhiều thị
    trường, chủ yếu là các cặp giao dịch với **USDT**, **BTC**, và **USDC**. Danh sách này
    bao gồm tổng cộng 106 cặp giao dịch, được phân loại theo các loại tài sản phổ
    biến như **Bitcoin (BTC)**, **Ethereum (ETH)**, các loại **altcoin** (ví dụ: **XRP**, **DOGE**,
    **AAVE**), và **stablecoin** (**USDT**, **USDC**) trong khoảng thời gian từ 2022 đến nay.
    """)
    # Đọc file CSV vào DataFrame
    metrics_df_combined = pd.read_csv("data/feature/metrics_combined_results.csv")

    # Thư mục chứa các file CSV
    csv_folder = "data/data_2022_2024"

    # Lấy danh sách các file CSV trong thư mục
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    # Lấy tên mã tiền (symbol) từ tên các file CSV
    symbols = [os.path.splitext(f)[0] for f in csv_files]

    # Cho phép người dùng chọn một symbol từ danh sách
    selected_symbol = st.selectbox("Chọn mã tiền", symbols)

    # Đọc dữ liệu từ file CSV tương ứng
    selected_file = os.path.join(csv_folder, selected_symbol + ".csv")
    df = pd.read_csv(selected_file)

    # Đảm bảo cột 'timestamp' được chuyển thành kiểu datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Lọc dữ liệu theo khoảng thời gian được chọn
    st.subheader("Lọc dữ liệu theo thời gian")
    start_date = st.date_input("Chọn ngày bắt đầu", value=df['timestamp'].min().date())
    end_date = st.date_input("Chọn ngày kết thúc", value=df['timestamp'].max().date())

    # Lọc DataFrame theo khoảng thời gian
    df_filtered = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

    # Hiển thị dữ liệu CSV sau khi lọc
    st.write(f"Dữ liệu của {selected_symbol} từ {start_date} đến {end_date}:")
    st.write(df_filtered)

    # Cho phép người dùng tùy chỉnh các chỉ số
    st.sidebar.subheader("Cài đặt chỉ báo")
    ma1_period = st.sidebar.number_input("Khoảng thời gian MA 1", min_value=1, max_value=100, value=20)
    ma2_period = st.sidebar.number_input("Khoảng thời gian MA 2", min_value=1, max_value=100, value=50)

    # Tính Moving Averages (MA)
    df_filtered[f'MA{ma1_period}'] = df_filtered['close'].rolling(window=ma1_period).mean()
    df_filtered[f'MA{ma2_period}'] = df_filtered['close'].rolling(window=ma2_period).mean()

    # Vẽ biểu đồ nến (Candlestick Chart) sử dụng Plotly
    st.subheader(f"Biểu đồ Nến (Candlestick) cho {selected_symbol}")

    # Tạo subplots chỉ với 1 hàng cho Candlestick Chart
    fig = make_subplots(
        rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=[f'Candlestick Chart for {selected_symbol}']
    )

    # Vẽ biểu đồ nến (Candlestick Chart)
    fig.add_trace(go.Candlestick(x=df_filtered['timestamp'],
                                open=df_filtered['open'],
                                high=df_filtered['high'],
                                low=df_filtered['low'],
                                close=df_filtered['close'],
                                increasing_line_color='green',
                                decreasing_line_color='red',
                                showlegend=False),
                  row=1, col=1)

    # Thêm Moving Averages vào biểu đồ Candlestick
    fig.add_trace(go.Scatter(x=df_filtered['timestamp'], y=df_filtered[f'MA{ma1_period}'], mode='lines', name=f'MA {ma1_period}', line=dict(color='blue', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df_filtered['timestamp'], y=df_filtered[f'MA{ma2_period}'], mode='lines', name=f'MA {ma2_period}', line=dict(color='orange', width=1)),
                  row=1, col=1)

    # Hiển thị biểu đồ nến
    st.plotly_chart(fig, use_container_width=True)

    # Thêm kiểu chữ và background
    st.markdown(
        """
        <style>
        body {
            background-color: #1e1e1e;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .streamlit-expanderHeader {
            font-family: 'Arial', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True
    )



elif tab_selection == "Phương pháp và chiến lược":
    
    # Tạo các nút bấm để chọn chiến lược
    PP_options = ["Momentum Cổ Điển",
                        "Regress Then Rank",
                        "Learning To Rank"]
    st.header("Lợi nhuận theo từng tháng")
    st.image("Profit_CR/Profit_month.png", caption="Lợi nhuận theo từng tháng")
    st.header("Cumulative Return từng chiến lược trong các phương pháp")
    selected_CR = st.radio("Chọn 1 trong 3: ", PP_options)
    if selected_CR == "Momentum Cổ Điển":
        st.subheader("Cumulative Return by Return")
        st.image("Profit_CR/CR_Return.png", caption="Cumulative Return by Return")
        st.subheader("Cumulative Return by MACD")
        st.image("Profit_CR/CR__MACD.png", caption="Cumulative Return by MACD")
    elif selected_CR == "Regress Then Rank":
        st.subheader("Cumulative Return by RandomForest")
        st.image("Profit_CR/CR_RandomForest.png", caption="Cumulative Return by RandomForest")
    elif selected_CR == "Learning To Rank":
        st.subheader("Cumulative Return by LambdaMART")
        st.image("Profit_CR/CR_lambda.png", caption="Cumulative Return by LambdaMART")
        st.subheader("Cumulative Return by ListNet")
        st.image("Profit_CR/CR_ListNet.png", caption="Cumulative Return by ListNet")
    # Các phần của Strategy
    st.header("Chỉ Số Chiến Lược Đầu Tư")
    metrics_df_combined = pd.read_csv('metrics/performance_metrics.csv')
  
    st.dataframe(metrics_df_combined)
   
        