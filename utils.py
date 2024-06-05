import os
import openai
from datetime import datetime
import streamlit as st
 
st.sidebar.markdown('[![](https://img.shields.io/badge/7조_소스코드_보러가기-red?logo=github)](https://github.com/int11/langchaine_medicine/blob/main/main.py)')

 
def configure_openai():  
    openai_api_key = st.sidebar.text_input(
        label="🗝️ OpenAI API Key",
        type="password",
        value=st.session_state['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in st.session_state else '',
        placeholder="sk-..."
        )
    
    
    if openai_api_key:
        st.session_state['OPENAI_API_KEY'] = openai_api_key
        os.environ['OPENAI_API_KEY'] = openai_api_key
    else:
        st.warning("🔑 API 키를 추가하고 진행해주세요!")
        st.info("🔗 링크를 통해 API 키를 발급받을 수 있습니다. https://platform.openai.com/account/api-keys")
        st.stop()

    model = "gpt-4o"
    try:
        client = openai.OpenAI()
        available_models = [{"id": i.id, "created":datetime.fromtimestamp(i.created)} for i in client.models.list() if str(i.id).startswith("gpt")]
        available_models = sorted(available_models, key=lambda x: x["created"])
        available_models = [i["id"] for i in available_models]

        model = st.sidebar.selectbox(
            label="✅ Model 선택",
            options=available_models,
            index=available_models.index(st.session_state['OPENAI_MODEL']) if 'OPENAI_MODEL' in st.session_state else 18  # 인덱스 18은 gpt-4o  
        )
        st.session_state['OPENAI_MODEL'] = model
    except openai.AuthenticationError as e:
        st.error(e.body["message"])
        st.stop()
    except Exception as e:
        print(e)
        st.error("문제가 발생한거 같아요, 잠시 후 다시 시도해보시겠어요?")
        st.stop()
    return model