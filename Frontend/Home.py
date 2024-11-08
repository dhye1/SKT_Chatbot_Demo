import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="🩺",
)

st.write("# 🩺EmoChat🩺 ")


st.sidebar.success("Select a demo above.")

st.markdown(
    """
    \n **👈 For a step-by-step diagnosis, select a conversation stage from the sidebar!**


    ### See more details..
    - **💊 Step 1: Emotion Recognition & Depression Diagnosis**

        We analyzes your responses through two modalities—audio and text—to diagnose your emotions. 
        
    - **💊 Step 2: Emotional Support Consultation**
    
        Based on the predicted emotions and depression symptom types, 
        this stage offers personalized counseling and tailored solutions to address your specific needs.
"""
)