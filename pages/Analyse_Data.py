import streamlit as st
from app import menu
from Config import Config
import pandas as pd
from models.BasicSPPModule import BasicSPPModule as BSPPM

st.set_page_config(
    page_title=Config.APP_NAME,
    page_icon=Config.APP_ICON
)
menu()

st.title('Phân tích các dữ liệu có thể xử lý & đưa ra dự đoán')

st.write("Các mã cổ phiếu phổ biến:")
st.markdown("""AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, KO, PEP, DIS, NFLX, INTC""")
stock_symbol = st.text_input("Nhập mã cổ phiếu", "AAPL")
period = st.selectbox("Chọn khoảng thời gian", ["3mo", "6mo", "1y", "2y", "5y"])
train_ratio = st.slider("Chọn tỷ lệ huấn luyện", min_value=0.5, max_value=0.9, value=0.7, step=0.05)

bspp = BSPPM(stock_symbol, period, train_ratio)
raw_data = bspp.fetch_data()
processed_data = bspp.prepare_data()

st.subheader('__Dữ liệu giá cổ phiếu được chọn trong khoảng thời gian qua:__')

st.write("**Dữ liệu gốc (raw data):**")
st.dataframe(raw_data)

st.subheader('Dữ liệu giá cổ phiếu được chọn sau khi lọc và bỏ qua các dữ liệu khác ngoài Close(giá đóng cửa):')

st.write("**Dữ liệu đã xử lý (processed data):**")
st.dataframe(raw_data[['Close']])

st.subheader('___Xây dựng mô hình hồi quy tuyến tính:___')

st.write("**Dự đoán của model Linear Regression(Cơ bản):**")
bspp.train_model()
bspp.predict()
bspp.visualize()

st.subheader('___Phân tích kết quả dự đoán:___')
st.write("**Kết quả dự đoán:**")
st.markdown(
  """
  1. Từ hình ảnh thực tế của biểu đồ & các chỉ số thống kê(MSE, MAE, R^2) ta thấy được mô hình dự đoán với độ sai lệch ***rất cao***.
  2. Mô hình không thể dự đoán chính xác giá cổ phiếu của công ty được chọn trong tương lai vì sai lệch so với giá trị thực tế.
  """
)
st.write("**Phương pháp cải thiện mô hình thống kê:**")
st.markdown(
  """
  ***Ta có thể nhận thấy việc xử lý dữ liệu chỉ gồm số liệu đã được thống kê từ trước & việc chỉ sử dụng 1 tập thống kê cơ bản đã gây hạn chế đối với việc phân tích & dự đoán hoàn toàn bằng mô hình hồi quy tuyến tính nên ta có thể tạo lập thêm các chỉ số mới từ những tập dữ liệu & số liệu thống kê đã bị bỏ qua để cải thiện.***
  1. Xét thêm đến các yếu tố khác như: Các chỉ báo kỹ thuật (Technical Indicators), các đặc trưng từ dữ liệu thời gian.
  2. Xây dựng mô hình hồi quy tuyến tính phức tạp hơn như: Mô hình hồi quy tuyến tính đa biến, Mô hình hồi quy tuyến tính với các đặc trưng mới.

  ==> Sử dụng các chỉ báo kỹ thuật để cải thiện mô hình hồi quy tuyến tính (RSI, MACD) thông qua kỹ thuật ___Time Series Analysis___.
  """
)