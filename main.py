import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader
from streaming import StreamHandler
import utils

st.title("당뇨 환자들을 위한 챗봇")

# openai key input gui. 없으면 여기서 멈춤 있으면 계속 진행
model_name = utils.configure_openai()


# qa_chain 최초로 한번 정의하고 session_state에 저장해둠.  os.environ['OPENAI_API_KEY'] 없는 상태로 Chroma 객체 생성하면 에러남
if "qa_chain" not in st.session_state:
    # 각 확장자 별로 문서 로더 정의
    documents = []

    loaders = {
        'pdf': {'loader':PyMuPDFLoader, 'kwargs': {}},
        'txt': {'loader':TextLoader, 'kwargs': {'autodetect_encoding': True}}
    }
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

    llm = ChatOpenAI(model_name=model_name, temperature=0, streaming=True)

    st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )


#chat gui
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="당뇨 관련 질문하세요!")

if user_query:
    with st.chat_message("user"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamHandler(st.empty())
        result = st.session_state["qa_chain"].invoke(
            {"query":user_query},
            {"callbacks": [st_cb]}
        )
        response = result["result"]
        st.session_state.messages.append({"role": "assistant", "content": response})

        # to show references
        for idx, doc in enumerate(result['source_documents'],1):
            filename = os.path.basename(doc.metadata['source'])
            _, ext = os.path.splitext(filename)
        
            if ext == '.txt':
                ref_title = f":blue[Reference {idx}: *{filename}*]"
            elif ext == '.pdf':
                page_num = doc.metadata['page']
                ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"

            with st.popover(ref_title):
                st.caption(doc.page_content)