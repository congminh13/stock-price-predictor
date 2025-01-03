import streamlit as st
from app import menu
from Config import Config
st.set_page_config(
      page_title=Config.APP_NAME,
      page_icon=Config.APP_ICON
  )
menu()
