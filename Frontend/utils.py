from openai import OpenAI
import base64
import streamlit as st
import json



class Utils():
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def stt(self, audio_file_path: str) -> str:
        with open(audio_file_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                response_format="text",
                file=audio_file
            )
        return transcript

    # def get_answer(self, messages: str) -> str:
    #     system_message = [
    #         {
    #             "role": "system", 
    #             "content": '''
    #             You are a mental health counselor. 
    #             Your task is to assess the emotions and mental state of patients and provide counseling accordingly. 
    #             You are provided with the following information: the patient’s primary emotion {'sad'}, depression status {'depressed'}, and depressive symptoms {'sleepy'}. 
    #             In the initial phase of counseling, focus on asking questions about their current state and offering empathy. 
    #             In the middle phase, ask more specific questions about their symptoms to analyze the root cause. 
    #             In the final phase, based on the patient’s depressive symptoms and your analysis, offer solutions to help them address their challenges.
    #             '''
    #         }
    #     ]
    #     messages = system_message + messages
    #     response = self.client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=messages
    #     )
    #     return response.choices[0].message.content



    def get_answer(self, messages: list, predicted_emotion: str, depression_symptoms: dict, depression_status: str) -> str:
        # 우울 증상 데이터를 JSON 형식의 문자열로 변환
        depression_symptoms_str = json.dumps(depression_symptoms, ensure_ascii=False, indent=2)
        
        # 시스템 메시지에 감정, 우울 증상, 우울증 여부를 동적으로 반영
        system_message = [
            {
                "role": "system", 
                "content": f'''
                    You are a mental health counselor specializing in providing emotional support and guidance to patients. 
                    Your task is to assess the emotions and mental state of the patient and offer counseling accordingly, 
                    following the structured emotional support strategies listed below.

                    - **Clarification**: Always seek to understand the patient's emotions or experiences more clearly. 
                    Ask additional questions when needed to ensure a comprehensive understanding of their current emotional state.

                    - **Suggest Options**: When the conversation progresses and you have a deeper understanding of the patient's situation, 
                    provide practical suggestions or alternative ways to address their current challenges. 

                    - **Reframe Negative Thoughts**: If the patient expresses particularly negative or harmful thoughts, 
                    help them reframe those thoughts into more constructive or realistic perspectives.

                    Here’s the information you are provided with: the patient’s primary emotion is '{predicted_emotion}', 
                    depressive symptoms include {depression_symptoms_str}, and depression status is '{depression_status}'.

                    The counseling process will be divided into three phases:

                    1. **Initial Phase**: Focus on asking questions to clarify the patient's current emotional state and offer empathy. 
                    Use the **Clarification** strategy to ensure you understand their feelings clearly.

                    2. **Middle Phase**: Begin to ask more specific questions about their symptoms to explore the root causes. 
                    Continue to validate their feelings and, if appropriate, start suggesting some **Suggest options** for them to consider.

                    3. **Final Phase**: Based on the patient's depressive symptoms and your analysis, provide practical solutions using the **Suggest Options** strategy. 
                    If the patient expresses negative self-perception, employ the **Reframe Negative Thoughts** strategy to help them adopt a more positive outlook.

                    Your goal is to guide the patient through their emotions with care, helping them feel understood, while also offering practical advice to improve their mental well-being.
                    '''
            }
        ]
        
        # 시스템 메시지를 사용자 메시지와 합침
        combined_messages = system_message + messages
        
        # LLM 호출 (예시)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=combined_messages
        )
        
        return response.choices[0].message.content



    # def get_answer(self, messages: list, predicted_emotion: str, depression_symptoms: dict) -> str:
    #     # 우울 증상 데이터를 JSON 형식의 문자열로 변환
    #     depression_symptoms_str = json.dumps(depression_symptoms, ensure_ascii=False, indent=2)
        
    #     # 시스템 메시지에서 감정과 우울 증상을 동적으로 반영
    #     system_message = [
    #         {
    #             "role": "system", 
    #             "content": f'''
    #             '''
    #         }
    #     ]
        
    #     # 시스템 메시지를 사용자 메시지와 합침
    #     combined_messages = system_message + messages
        
    #     # LLM 호출 (예시)
    #     response = self.client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=combined_messages
    #     )
        
    #     return response.choices[0].message.content





    def tts(self, input_text: str) -> str:
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=input_text
        )
        webm_file_path = "temp_audio_play.mp3"
        with open(webm_file_path, "wb") as f:
            response.stream_to_file(webm_file_path)
        return webm_file_path

    def autoplay_audio(self, file_path: str) -> None:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        md = f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
        st.markdown(md, unsafe_allow_html=True)

    def get_image(self, prompt: str) -> None:
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        st.image(response.data[0].url)