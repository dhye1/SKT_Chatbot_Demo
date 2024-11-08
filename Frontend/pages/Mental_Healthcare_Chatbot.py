############################### 2. 예측값 기반 mental healthcare page ###############################
import os
from dotenv import load_dotenv
import streamlit as st
from src.utils import Utils

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("openai_api_key")
utils = Utils(api_key=api_key)

# 메시지 세션 초기화 함수
def reset_session():
    st.session_state.messages = []
    st.session_state.turn_count = 0  # 대화 횟수 초기화
    
    # 초기 메시지 설정
    predicted_emotion = st.session_state.get("predicted_emotion", "sad")
    depression_status = "**No** signs of **depression** are detected." if st.session_state.get("depression_status") == "non" else "Signs of **depression** are detected."
    first_message = (
        f"It seems you're currently experiencing a **{predicted_emotion} emotion**. "
        f" {depression_status}"
        " Could you share more about any recent experiences that might be contributing to these feelings?"
    )

    
    # 첫 번째 메시지를 세션에 추가
    st.session_state.messages.append({"role": "assistant", "content": first_message})

st.write("### Mental Healthcare Chatbot 🩺")

# Reset Session 버튼 처리
if st.button("Reset Session", key="reset_button"):
    reset_session()

# 초기 메시지 설정 (세션 상태에 messages가 없을 경우에만 초기화)
if "messages" not in st.session_state:
    reset_session()

# 모든 대화 기록을 순서대로 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력 및 처리 함수
def on_input_submit():
    user_input = st.session_state.user_input
    if user_input:
        # 유저 입력을 메시지로 저장하고 즉시 화면에 출력
        st.session_state.messages.append({"role": "user", "content": user_input})
        # with st.chat_message("user"):
        #     st.write(user_input)

        # 입력 필드 초기화
        st.session_state.user_input = ""

        # 채팅 반복 카운트 증가
        st.session_state.turn_count += 1



        # 사용자 입력 후, spinner를 출력하여 챗봇의 응답 대기 상태를 표시
        with st.spinner("Thinking... 🤔"):
            final_response = None
            depression_status = "No" if st.session_state.get("depression_status") == "non" else "Yes"
            if st.session_state.turn_count < 5:
                final_response = utils.get_answer(
                    messages=st.session_state.messages,
                    predicted_emotion=st.session_state.get("predicted_emotion"),
                    depression_symptoms=st.session_state.get("depression_symptoms"),
                    depression_status=depression_status  # 추가된 인자
                )
            else:
                final_response = "Thank you for sharing. Feel free to reach out if you need more support!"

        
        # spinner가 끝난 후 챗봇 응답을 화면에 출력
        if final_response:
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            # with st.chat_message("assistant"):
            #     st.write(final_response)


st.text_input("Enter your message...", key="user_input", on_change=on_input_submit)






