import fitz  # PyMuPDF

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.chains import RetrievalQA
import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
# 시스템 프롬프트 정의
SYS_PROMPT = "You are a chatbot designed to help diabetes patients. Provide answers based on the provided context."

# 프롬프트 템플릿
template = SYS_PROMPT + '''Answer the question based only on the following context:
{context}

Question: {question}
'''

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_template(template)

# 각 확장자 별로 문서 로더 정의
loaders = {
    'pdf': {'loader':PyMuPDFLoader, 'kwargs': {}},
    'txt': {'loader':TextLoader, 'kwargs': {'autodetect_encoding': True}}
}

documents = []
for file_type, value in loaders.items():
    loader = value['loader']
    loader_kwargs = value['kwargs']

    loader = DirectoryLoader(path=f"data/{file_type}", glob=f"**/*.{file_type}",loader_cls=loader, loader_kwargs=loader_kwargs)
    documents.extend(loader.load())

# 간단한 키워드 기반 문서 검색기 정의
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding)

retriever = vectordb.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model='gpt-3.5-turbo', temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True)


# 스트림릿 앱
st.title("당뇨 환자들을 위한 챗봇")
st.write("당뇨 관련 질문을 입력하세요:")

question = st.text_input("질문:")

if question:
    response = qa_chain.invoke(question)
    st.markdown(f"**답변:** {response}")