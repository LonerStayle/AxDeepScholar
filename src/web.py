import streamlit as st
import requests

st.title("🧠 챗봇")

API_URL = "http://localhost:8000/run"

if "messages" not in st.session_state:
    st.session_state.messages = []


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_input = st.chat_input("질문을 입력하세요")

if user_input:
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    
    with st.spinner("⏳ 응답 생성 중..."):
        res = requests.post(API_URL, json={"messages": st.session_state.messages}).json()

    
    final_msg = res["messages"][-1] if res.get("messages") else "응답 없음"
    st.session_state.messages.append({"role": "assistant", "content": final_msg})
    with st.chat_message("assistant"):
        st.write(final_msg)

    st.rerun()