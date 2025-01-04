import streamlit as st
from app import menu
from Config import Config
st.set_page_config(
    page_title=Config.APP_NAME,
    page_icon=Config.APP_ICON
)
menu()

from models.ImprovedSPPModule import ImprovedSPPModule as ISPPM

st.title('Cách triển khai mô hình cải tiến')

st.markdown(
  """
  Nhận thấy việc có thể sử dụng các chỉ số thống kê cơ bản để dự đoán giá cổ phiếu của công ty trong tương lai, ta có thể thiết lập công thức để sử dụng và xây dựng thêm các tập dữ liệu mới cho mô hình hồi quy tuyến tính.
  """
)

# RSI Equation
st.subheader("___RSI (Relative Strength Index):___")
st.markdown("RSI (Relative Strength Index) là một chỉ báo kỹ thuật phổ biến, được sử dụng để đo lường sức mạnh tương đối của giá cổ phiếu qua các chu kỳ gần đây.")
st.latex(r"""
\text{RSI} = 100 - \frac{100}{1 + RS} , \quad \text{với } RS = \frac{\text{Tăng trung bình}}{\text{Giảm trung bình}}
""")

st.markdown("___Thiết lập mã:___")
st.code(
    """
    @staticmethod
    def calculate_rsi(data, period=14):
      delta = data['Close'].diff(1)
      gain = delta.where(delta > 0, 0)
      loss = -delta.where(delta < 0, 0)
      avg_gain = gain.rolling(window=period).mean()
      avg_loss = loss.rolling(window=period).mean()
      rs = avg_gain / avg_loss
      rsi = 100 - (100 / (1 + rs))
      return rsi
    """,
    language="python"
)

# MACD Equation
st.subheader("***MACD (Moving Average Convergence Divergence):***")
st.markdown("MACD (Moving Average Convergence Divergence) là một chỉ báo dao động, giúp đánh giá mối quan hệ giữa hai đường trung bình động theo thời gian.")
st.latex(r"""
\text{MACD} = \text{EMA}_{\text{ngắn hạn}} - \text{EMA}_{\text{dài hạn}} , \quad \text{Đường tín hiệu} = \text{EMA}_{\text{kỳ tín hiệu}}(\text{MACD})
""")

st.markdown("___Thiết lập mã:___")
st.code(
    """
    @staticmethod
    def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
      short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
      long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
      macd = short_ema - long_ema
      signal_line = macd.ewm(span=signal_window, adjust=False).mean()
      return macd, signal_line
    """,
    language="python"
)
st.subheader("***Xây dụng mô hình:***")
st.markdown(
  """
    ***Như vậy, ta có thể xây dựng mô hình hồi quy tuyến tính mới với 2 tập dữ liệu mới là RSI & MACD để cải thiện mô hình hồi quy tuyến tính cơ bản.***
  """
)
st.write("Các mã cổ phiếu phổ biến:")
st.markdown("""AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, KO, PEP, DIS, NFLX, INTC""")
stock_symbol = st.text_input("Nhập mã cổ phiếu", "AAPL")
period = st.selectbox("Chọn khoảng thời gian", ["3mo", "6mo", "1y", "2y", "5y"])
train_ratio = st.slider("Chọn tỷ lệ huấn luyện", min_value=0.5, max_value=0.9, value=0.7, step=0.05)

isppm = ISPPM(stock_symbol, period, train_ratio)
isppm.fetch_data()
isppm.prepare_data()
isppm.train_model()
isppm.predict()
isppm.visualize()