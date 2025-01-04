import os
import streamlit as st
from Config import Config

def menu():
  st.sidebar.page_link('app.py', label='Trang chính')
  st.sidebar.page_link('pages/Analyse_Data.py', label='Phân tích dữ liệu')
  st.sidebar.page_link('pages/Customization.py', label='Xây dựng mô hình cải tiến')
  st.sidebar.page_link('pages/Evaluation.py', label='Đánh giá mô hình')


if __name__ == '__main__':
  st.set_page_config(
      page_title=Config.APP_NAME,
      page_icon=Config.APP_ICON
  )

  st.title(f'{Config.APP_NAME}')
  st.write('Version:‎ ‎ ‎ ‎', Config.APP_VERSION)
  st.markdown('Trần Công Minh  -  CSA08')

  st.divider()
  
  st.subheader('Ý tưởng:')
  st.markdown(
      """
        <p style="font-size:18px; text-align:justify;">
        Phân tích độ hiệu quả & tính khả thi của mô hình hồi quy tuyến tính(Linear Regression) trong việc dự đoán giá cổ phiếu & xây dựng mô hình cải tiến.
        </p>
        """,
      unsafe_allow_html=True
    )
  st.subheader('Triển khai:')
  st.markdown(
    """
      <ul style="font-size:18px; text-align:justify; list-style-type:disc; line-height:32px;">

        1. Lựa chọn khoảng thông tin cụ thể(thời gian, mã cổ phiếu) để thu thập dữ liệu giá cổ phiếu của công ty.
        2. Thu thập dữ liệu: Dữ liệu giá cổ phiếu của công ty được chọn (sử dụng API Yahoo Finance (yfinance)) trong khoảng thời gian được chọn.
        3. Tiền xử lý dữ liệu: Loại bỏ dữ liệu thiếu, chuẩn hóa dữ liệu, chia dữ liệu thành tập huấn luyện và tập kiểm tra.
        4. Xây dựng & sử dụng mô hình mô hình hồi quy tuyến tính để dự đoán giá cổ phiếu.
        5. Đánh giá mô hình thông qua các đánh giá mô hình.
      </ul>
    """,
    unsafe_allow_html=True
  )

  menu()
  


  
