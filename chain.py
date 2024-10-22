

import os
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import faiss

from langchain import PromptTemplate
from langchain import hub
from langchain.docstore.document import Document

from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain.vectorstores import FAISS
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

from uuid import uuid4
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os


class JejuRestaurantRecommender:
    def __init__(self, vector_store_path):
        # 디바이스 설정 (cuda 사용 가능 여부 확인)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Hugging Face의 사전 학습된 임베딩 모델과 토크나이저 로드
        self.model_name = "intfloat/multilingual-e5-large-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': False},
            show_progress=True
        )
        
        # 벡터 DB 로드
        self.vector_store_path = './vectorstore'
        self.db_meta = FAISS.load_local(self.vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)
        
        # 리트리버 설정
        self.retriever = self.db_meta.as_retriever()
        
        # 프롬프트 템플릿 설정
        self.llm_prompt_raw = PromptTemplate.from_template(
            """
            당신은 제주 맛집 추천 전문가입니다.  \
            당신은 사용자 질문에 맞춰 맛집을 추천해주세요. 추천할 때, 제주도 맛집 메타 데이터를 기준으로 추천해주세요.
            메타 데이터에는 아래와 같은 정보들을 포함하고 있습니다.

            가맹점명 : 맛집 가게 이름
            가맹점업종 : 가맹점 카테고리에 대한 내용 (단품요리, 한식, 일식 등)
            가맹점주소 : 가맹점이 위치한 주소
            이용건수 : 해당 가맹점의 이용건수로 2023년도 월별 이용건수에서 최빈도 구간으로 적용한 기준
            이용금액 : 해당 가맹점의 이용금액으로 2023년도 월별 이용건수에서 최빈도 구간으로 적용한 기준
            건당평균이용금액 : 해당 가맹점의 건당평균이용금액으로 2023년도 월별 이용건수에서 최빈도 구간으로 적용한 기준
            요일별 이용건수 비중 : 월요일부터 일요일까지 요일의 이용건수 비중
            시간대별 이용건수 비중 : 오전 5시부터 다음날 오전 4시까지 시간대의 이용건수 비중
            현지인이용건수 비중 : 현지인 이용건수 비중
            남성과 여성 성별 비중 : 남성, 여성 회원수 비중
            연령대별 회원수 비중 : 20대, 30대, 40대, 50대, 60대 회원수 비중

            \

            답변 할 때, "선택하신 지역, 성별, 나이에 해당하는 제주도 맛집 정보에 의하면.." 으로 시작하여 답변해주세요
            '추천이유' 항목에는 어떤 근거로 해당 가게가 검색되고 추천되었는지 설명해주세요.
            추천된 가게가 여러개인 경우, 여러가게를 리스팅해서 응답해주세요
            사용자 질문에 대한 응답 :{context}


            질문: {question}
            답변:
            
            \n- 가게명 : ** 가맹점명**
            \n
            \n- 업종 :
            \n- 주소 :
            \n- 추천 이유 : 
            \n- 참고) 사용한 데이터의 이용건수, 이용금액, 건당 평균 이용금액은 2023년도 월별 이용건수에서 최빈도 구간으로 적용한 기준이며, 
            \n  요일/시간대/현지인 이용건수 비율, 성별/나이대는 2023년도 월별  수치를 평균화하여 적용한 기준입니다.

            """ 
        )

            # \n- 이용건수 구간(최빈도):
            # \n- 이용금액 구간(최빈도):
            # \n- 건당 평균 이용금액(최빈도):
            # \n- 요일별 이용건수 비중:
            # \n- 시간대별 이용건수 비중:
            # \n- 현지인 이용건수 비중:
            # \n- 성별 비중 :
            # \n- 연령대 비중 :





        # Gemini 설정
        self.llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def create_meta_chain(self):
        # 메타 체인 구성
        meta_chain = (
            {
                'context': lambda x: (
                    result := self.retriever.invoke(x["question"]),
                    self.format_docs(result),
                    #print(f"meta_chain retriever 검색 결과: {result}")  # 검색 결과를 출력
                )[1],  # format_docs가 2 번째 항목이므로 [1]
                'question': lambda x: x["question"]
            }
            | self.llm_prompt_raw
            | self.llm_model
            | StrOutputParser()
        )
        return meta_chain

    def search(self, query):
        # 메타 체인을 생성하고 질문을 처리하여 응답을 반환
        meta_chain = self.create_meta_chain()
        res = meta_chain.invoke({"question": query})
        #print(res)
        return res
    
    
    
# 프롬프트
# llm_prompt_recomm = PromptTemplate.from_template(
#             """"
#             당신은 제주 맛집 추천 전문가입니다.  \
#             당신은 사용자 질문에 맞춰 맛집을 추천해주세요. 추천할 때, 제주도 맛집 추천 데이터를 기준으로 추천해주세요.
#             추천 데이터에는 아래와 같은 정보들을 포함하고 있습니다.

#             가게명 : 맛집 가게 이름
#             업종 : 가게 카테고리에 대한 내용 (단품요리, 한식, 일식 등)
#             주소 : 가맹점이 위치한 주소
#             운영시간 : 가게의 운영시간 
#             번호 : 가게 전화번호 
#             부가설명 : 포장, 단체, 예약 등 가게 관련 부가정보 
#             링크 : 가게 링크 URL
#             메뉴 : 가게 메뉴
#             가게 클러스터링 속성 : 가게 음식이 속하는 주요 클러스터링 속성

#             \
            
#             답변 할 때, "선택하신 지역, 성별, 나이에 해당하는 제주도 맛집 추천 정보에 의하면.." 으로 시작하여 답변해주세요
#             '추천이유' 항목에는 어떤 근거로 해당 가게가 검색되고 추천되었는지 설명해주세요.
#             추천된 가게가 여러개인 경우, 여러가게를 리스팅해서 응답해주세요
#             사용자 질문에 대한 응답 :{context}


#             질문: {question}
#             답변:
#             \n- 가게명 : ** 가맹점명**
#             \n
#             \n- 업종 :
#             \n- 주소 :
#             \n- 운영시간 :
#             \n- 번호 :
#             \n- 부가설명 :
#             \n- 링크 :
#             \n- 메뉴 :
#             \n- 추천이유 :                                    
#             """ )
