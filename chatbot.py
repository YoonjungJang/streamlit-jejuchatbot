import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import faiss

from function import *

import streamlit as st

from chain import JejuRestaurantRecommender

# 경로 설정
#data_path = './data'
#module_path = './modules'

# Gemini 설정
import google.generativeai as genai

# import shutil
# os.makedirs("/root/.streamlit", exist_ok=True)
# shutil.copy("secrets.toml", "/root/.streamlit/secrets.toml")

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)



# Gemini 모델 선택
model = genai.GenerativeModel("gemini-1.5-flash")


st.set_page_config(page_title="Jeju food chatbot", page_icon="🤖",layout="wide")

# Replicate Credentials
with st.sidebar:
    st.title("🤖제주도 맛집 추천 챗봇")

    st.write("")

    st.subheader("제주도 지역")

    # selectbox 레이블 공백 제거
    st.markdown(
        """
        <style>
        .stSelectbox label {  /* This targets the label element for selectbox */
            display: none;  /* Hides the label element */
        }
        .stSelectbox div[role='combobox'] {
            margin-top: -20px; /* Adjusts the margin if needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    area = st.sidebar.selectbox("", ["제주시", "서귀포시", "동제주군", "서주제군"], key="area")

    st.write("")

    st.subheader("성별")

    # radio 레이블 공백 제거
    st.markdown(
        """
        <style>
        .stRadio > label {
            display: none;
        }
        .stRadio > div {
            margin-top: -20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    gender = st.radio(
        '',
        ('남성', '여성')
    )

    st.write("")
    
    st.subheader("나이")

    # radio 레이블 공백 제거
    st.markdown(
        """
        <style>
        .stRadio > label {
            display: none;
        }
        .stRadio > div {
            margin-top: -20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    age = st.radio(
        '',
        ('20대 이하', '30대', '40대', '50대', '60대')
    )
    
    
    
# chat-bot 페이지 디자인
st.title(f"🍊취향저격 제주도 맛집 추천 챗봇")

st.write("뜨끈한 국물?🍜 매콤한 음식?🌶️ 현재 어떤 음식을 먹고 싶은지 알려주세요")

#st.write("제주도 지도에서 구역을 클릭하여 지역 맞춤형 맛집을 추천받으세요.")
st.write("제주도 지역 이미지를 참고하여 좌측에 지역을 선택해주세요")

st.write("")

# image_path = "./jejumap.png"
# image_html = f"""
# <div style="display: flex; justify-content: center;">
#     <img src="{image_path}" alt="centered image" width="50%">
# </div>
# """
# st.markdown(image_html, unsafe_allow_html=True)

# st.write("")
from PIL import Image
# 이미지 불러오기
image = Image.open("./jejumap.png")

# 이미지 표시
# st.image(image, use_column_width=True)
st.image(image, width=400)
# st.markdown("""
# <style>
# image {
# 	max-height: 500px;
# }
# </style>
# """, unsafe_allow_html=True)



# chatbot 입출력
chain=JejuRestaurantRecommender('./vectorstore')


def is_place_related_question(user_input):
    keywords = ["가게", "레스토랑", "식당", "추천", "음식점", "장소", "여기", "위치","곳","알려줘","찾아줘","말해줘"]
    return any(keyword in user_input for keyword in keywords)

def generate_answer(user_input):
    res=chain.search(user_input)
    
    return res


#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("안녕하세요 👋 어떤 맛집을 추천해드릴까요?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    gmap_id=""
    opening_talk=["안녕","안녕하세요","하이","안녕?","안녕하세요?","하이?","안녕하세요!"]
    
    if prompt in opening_talk: res="안녕하세요 😊 어떤 맛집을 추천해드릴까요?"
    elif is_place_related_question(prompt):
        res=generate_answer(prompt)
        print("res:",res)
        
    else:
        res=" 먹고 싶은 음식을 말해주세요🤔"
       
    
        
    st.session_state.messages.append({"role": "assistant", "content": res})
    
    with st.chat_message("assistant"): 
        with st.spinner("Thinking..."):
            st.write_stream(stream_data(res))
    
    
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어떤 맛집을 추천해드릴까요?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)    
