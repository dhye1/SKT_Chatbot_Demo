###################### 감정 및 우울 증상 예측 page ###############################
import os
import requests
import subprocess
from dotenv import load_dotenv
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
from src.utils import Utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import transformers
import streamlit.components.v1 as components

load_dotenv()
api_key = os.getenv("openai_api_key")
utils = Utils(api_key=api_key)

# 이미지 파일 경로 설정
IMAGE_PATHS = {
    "anger": "img/anger.png",
    "excited": "img/excited.png",
    "frustrated": "img/frustrated2.png",
    "happy": "img/happy.png",
    "neutral": "img/neutral.png",
    "sadness": "img/sadness2.png",
}

# Flask 기반 멀티모달 모델이 돌아가는 서버의 주소와 포트 설정
FLASK_API_URL = "http://localhost:5000/predict"

# BERT 모델과 토크나이저 로드 (감정 예측용)
tokenizer = transformers.AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion", use_fast=True)
model = transformers.AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion").cuda()

# 예측을 위한 파이프라인 생성
pred = transformers.pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0,
    return_all_scores=True,
)

# Reset session state for messages when entering this page
def reset_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "audio_data" not in st.session_state:
        st.session_state["audio_data"] = b""  # 오디오 데이터를 저장할 변수
    if "full_transcript" not in st.session_state:
        st.session_state["full_transcript"] = ""  # 전체 텍스트를 저장할 변수
    st.session_state["prediction_complete"] = False  # 예측 완료 플래그 초기화
    st.session_state["final_message"] = ""  # 최종 메시지 초기화
    st.session_state["displayed_messages"] = []  # 이미 표시된 메시지를 추적

reset_session_state()
float_init()

st.write("### Emotion Recognition & Depression Detection 🩺")
st.info("마이크 버튼을 누르고 마이크에 말해주세요.")

questions = [
    "How have you been over the past few days? Have you felt any particular feelings recently?",
    "Thank you for sharing, it sounds like this has been meaningful for you. Have there been any specific events or situations that have affected your mood lately?",
    "When you feel this way, what kinds of thoughts usually come to mind?",
    "Have you been experiencing any changes in appetite, sleep, or energy levels recently?",
    "Would you say these feelings have been consistent, or do they come and go throughout the day?"
]

with st.chat_message("assistant"):
    st.write(questions[0])

def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": questions[0]}]
    if "question_idx" not in st.session_state:
        st.session_state.question_idx = 0

initialize_session_state()

# 첫 번째 질문을 강제로 화면에 출력
if st.session_state.messages and not st.session_state.displayed_messages:
    with st.chat_message("assistant"):
        st.write(st.session_state.messages[0]["content"])
    st.session_state.displayed_messages.append(st.session_state.messages[0])

# 이전 대화 내용 출력 (이미 표시된 메시지만 출력)
for message in st.session_state.messages[1:]:
    if message not in st.session_state.displayed_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
        st.session_state.displayed_messages.append(message)

footer_container = st.container()
with footer_container:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<p style='font-size: 24px; font-weight: bold; margin-left: 170px;'>Click to record</p>", unsafe_allow_html=True)
    with col2:
        audio_bytes = audio_recorder(icon_name="microphone", icon_size="3x", neutral_color="#000000", recording_color="#fc0000", text="")

# 음성 데이터 처리
if audio_bytes:
    with st.spinner("음성 인식 중..."):
        try:
            # webm 파일로 저장
            webm_file_path = "temp_audio.webm"
            with open(webm_file_path, "wb") as f:
                f.write(audio_bytes)

            # ffmpeg를 사용해 webm 파일을 wav로 변환
            wav_file_path = "temp_audio.wav"
            ffmpeg_command = ["ffmpeg", "-y", "-i", webm_file_path, wav_file_path]
            subprocess.run(ffmpeg_command, check=True)

            transcript = utils.stt(wav_file_path)
            if transcript:
                # 유저 메시지를 세션에 추가하고 full_transcript에 줄바꿈으로 추가
                st.session_state.messages.append({"role": "user", "content": transcript})
                st.session_state.full_transcript += f"{transcript}\n"  # 'User:' 생략하고 한 줄씩 추가
                with st.chat_message("user"):
                    st.write(transcript)

            # 질문 추가
            question_idx = st.session_state.question_idx
            if question_idx < len(questions) - 1:
                question_idx += 1
                st.session_state.question_idx = question_idx
                next_question = questions[question_idx]

                # Assistant 메시지를 세션에 추가 (full_transcript에는 추가하지 않음)
                st.session_state.messages.append({"role": "assistant", "content": next_question})
                with st.chat_message("assistant"):
                    st.write(next_question)
            else:
                with st.spinner("전체 대화 세션에 대한 감정 및 우울증 예측 중..."):
                    with open(wav_file_path, "rb") as audio_file:
                        files = {'audio_file': audio_file}
                        data = {'text': st.session_state.full_transcript}  # User 메시지만 포함된 full_transcript 전송
                        try:
                            response = requests.post(FLASK_API_URL, files=files, data=data, timeout=30)
                            if response.status_code == 200:
                                response_data = response.json()
                                emotion_prediction = response_data.get("emotion_prediction", "예측 실패")
                                depression_prediction = response_data.get("depression_prediction", "예측 실패")
                                symptom_weights = response_data.get("symptom_weights", {})

                                # 예측 결과 저장
                                st.session_state["predicted_emotion"] = emotion_prediction
                                st.session_state["depression_symptoms"] = symptom_weights
                                st.session_state["depression_status"] = depression_prediction

                                # 예측 결과 메시지
                                final_message = f"Your current emotion is **'{emotion_prediction}'**."
                                st.session_state["final_message"] = final_message
                                st.session_state.messages.append({"role": "assistant", "content": final_message})

                                # 감정 예측 결과에 따른 이미지 출력
                                # st.write("### Overall predictions")
                                image_path = IMAGE_PATHS.get(emotion_prediction)
                                if image_path and os.path.exists(image_path):
                                    st.image(image_path, caption=f"Emotion: {emotion_prediction.capitalize()}", use_column_width=True)
                                else:
                                    st.write(f"Image for {emotion_prediction} not found.")

                                # 중복 제거 및 한 줄씩 띄워서 저장한 user_transcript 생성
                                user_messages = [message["content"] for message in st.session_state.messages if message["role"] == "user"]

                                # 중복을 제거하고 줄바꿈으로 연결
                                unique_user_messages = []
                                for msg in user_messages:
                                    if msg not in unique_user_messages:
                                        unique_user_messages.append(msg)

                                user_transcript = "\n".join(unique_user_messages)

                                # SHAP 해석 생성 및 시각화 결과 표시 (HTML 임베드)
                                explainer = shap.Explainer(pred)
                                shap_values = explainer([user_transcript])
                                shap_html = shap.plots.text(shap_values[0], display=False)
                                components.html(shap_html, height=200)


                                st.session_state["prediction_complete"] = True
                            else:
                                st.error(f"예측 실패 - 상태 코드: {response.status_code}, 내용: {response.text}")

                        except requests.exceptions.Timeout:
                            st.error("서버 응답 시간이 초과되었습니다.")
                        except requests.exceptions.RequestException as e:
                            st.error(f"요청 중 오류 발생: {e}")

            os.remove(webm_file_path)
            os.remove(wav_file_path)

        except Exception as e:
            st.error(f"오류 발생: {str(e)}")




# 히트맵 시각화 함수
def plot_symptom_heatmap(symptom_weights):
    symptoms = list(symptom_weights.keys())
    weights = list(symptom_weights.values())
    data = pd.DataFrame(weights, index=symptoms, columns=["Weight"])

    plt.figure(figsize=(3, 2))
    sns.heatmap(
        data,
        annot=True,
        cmap="PuBuGn",  # 노란색에서 빨간색으로의 그라데이션 컬러맵
        cbar=True,
        linewidths=0.5,
        fmt=".4f",
        vmin=0,
        vmax=1  # 최대 값을 1로 설정하여 강한 색 대비 적용
    )
    st.pyplot(plt)


# 예측 완료 시 최종 메시지와 히트맵을 출력
if st.session_state.get("prediction_complete"):
    # 이미 출력된 메시지와 중복되지 않도록 관리
    for message in st.session_state.messages:
        if message not in st.session_state.displayed_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
            st.session_state.displayed_messages.append(message)

    # 최종 메시지가 이미 메시지 리스트에 있는지 확인 후 추가
    final_message = f"Your current emotion is **'{st.session_state['predicted_emotion']}'**."
    if {"role": "assistant", "content": final_message} not in st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": final_message})

    # # 전체 대화 내용 (full_transcript) 출력
    # st.write("### Full Transcript")
    # st.write(st.session_state["full_transcript"])  # full_transcript 출력

    # 감정 예측 결과 출력
    st.write("### Prediction Results")
    st.write(f"**Emotion Prediction**: {st.session_state['predicted_emotion']}")

    # 우울증 여부 출력 ('non'이면 No, 그 외에는 Yes)
    depression_status = "No" if st.session_state.get("depression_status") == "non" else "Yes"
    st.write(f"**Depression Status**: {depression_status}")

    # 히트맵 출력
    symptom_weights = st.session_state["depression_symptoms"]
    plot_symptom_heatmap(symptom_weights)



footer_container.float("bottom: 0rem;")








