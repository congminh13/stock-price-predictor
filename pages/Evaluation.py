import streamlit as st
from app import menu
from Config import Config
st.set_page_config(
  page_title=Config.APP_NAME,
  page_icon=Config.APP_ICON,
  layout="wide"
)
menu()

from models.CombinedGraph import CombinedGraph as CG

st.title('Đánh giá & Kết luận')

st.subheader('***So sánh 2 mô hình hồi quy tuyến tính:***')

st.write("Các mã cổ phiếu phổ biến:")
st.markdown("""AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, KO, PEP, DIS, NFLX, INTC""")
stock_symbol = st.text_input("Nhập mã cổ phiếu", "AAPL")
period = st.selectbox("Chọn khoảng thời gian", ["3mo", "6mo", "1y", "2y", "5y"])
train_ratio = st.slider("Chọn tỷ lệ huấn luyện", min_value=0.5, max_value=0.9, value=0.7, step=0.05)

cg = CG(stock_symbol, period, train_ratio)
cg.visualize()

st.subheader('***Đánh giá:***')
st.markdown(
  """
    1. Từ việc so sánh giữa 2 mô hình hồi quy tuyến tính cơ bản & cải tiến, ta thấy được mô hình cải tiến có độ chính xác cao hơn ***rất nhiều*** so với mô hình cơ bản.
    2. Mô hình cải tiến có thể dự đoán chính xác giá cổ phiếu của công ty được chọn trong tương lai với sai số ***thấp hơn***.
    3. Việc tối ưu hóa mô hình thông qua việc xử lý dữ liệu, thêm các chỉ số thống kê mới đã giúp cải thiện đáng kể hiệu suất của mô hình hồi quy tuyến tính.
    4. Việc nắm giữ nhiều dữ liệu thống kê là rất quan trọng nhưng việc xử lý thông tin đó để tạo ra một mô hình dự đoán chính xác là điều ưu tiên cần thiết hơn cả.
  """
)

st.subheader('***Kết luận:***')
st.markdown('***Như vậy, việc sử dung mô hình hồi quy tuyến tính(Linear Regression) là hoàn toàn khả thi nếu ta biết tận dụng các dữ liệu thống kê thuần một cách hiệu quả và thiết lập tối ưu mô hình đó.***')
st.markdown('***Lưu ý:*** mô hình cải tiến được sử dụng chưa thật sự tối ưu vì không tận dụng hết các chỉ số thống kê có thể có từ dữ liệu cổ phiếu và đây chỉ là mẫu để trả lời câu hỏi của đề tài.')