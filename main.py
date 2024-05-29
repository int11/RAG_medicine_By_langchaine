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

os.environ["OPENAI_API_KEY"] = ""
# 시스템 프롬프트 정의
SYS_PROMPT = "You are a chatbot designed to help diabetes patients. Provide answers based on the provided context."

# 프롬프트 템플릿
template = SYS_PROMPT + '''Answer the question based only on the following context:
{context}

Question: {question}
'''

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_template(template)


# 문서 형식화 함수 정의
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# PDF에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# PDF 파일 경로
pdf_path = "2023_당뇨병_진료지침_전문_최종.pdf"  # 실제 PDF 파일 경로로 교체

# PDF에서 추출한 텍스트를 문서로 변환
pdf_text = extract_text_from_pdf(pdf_path)
documents = [Document(page_content=pdf_text)]

# 간단한 키워드 기반 문서 검색기 정의
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding)

retriever = vectordb.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model='gpt-4o', temperature=0),
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