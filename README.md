## SKT AI Fellowship - Multimodal Project

## Project demo
This repository includes the code and demo of our multimodal chatbot system.

## Folder Architecture
All model checkpoints are separately stored in the [Dropbox](https://www.dropbox.com/scl/fo/eij3dnkeccbyzuvc4qavm/AOs9R-huka-AIwXPsvd_0Ao?rlkey=tn5hqll9arnunbyz8odu3sz5y&st=59t5deol&dl=0).
### **Backend**
Backend folder contains server-side code for the chatbot system, implemented using Flask. This folder includes files that enable multimodal emotion and depression prediction using audio and text inputs.

1. flask.py
   - **Model Loading Functions**:
     - `load_emotion_model()`: Initializes and loads weights for the emotion prediction model.
     - `load_depression_model()`: Loads weights and settings for the depression prediction model.
   - **Prediction Functions**:
     - `predict_emotion()`: Processes audio and text inputs to predict an emotional label.
     - `predict_depression_symptoms()`: Processes audio and text inputs to predict a depression label and symptom weights.
   - **API Endpoint**:
     - `/predict`: Accepts audio and text input, predicts emotion and depression states, and returns results in JSON format.


2. wav2vec.py
   - This file contains code to create a customized wrapper around the wav2vec model, designed for emotion prediction tasks.

3. wavlm_plus.py
   - It includes two main encoder layers and a wrapper in wavlm, which integrates model with additional layers and settings.


4. whisper_model.py
   - The whisper_model.py file provides a custom wrapper for the Whisper model, integrating LoRA.
     
5. adapter.py
   - It includes Adapter class, a parameter-efficient fine-tuning module designed to add adaptable layers to model.
     
6. custom_roberta.py
   - This file contains a customized implementation of the Roberta model with added support for cross-modal attention.
     
7. prediction.py
   - This file contains a `TextAudioClassifier` class, which is a multimodal classification model that combines audio, text, and speaker information to predict across different modalities.
     
8. depression_prediction.py
   - This file contains a `TextAudioClassifier` class, integrating audio, text, and speaker information with specific configurations for PHQ symptom and depression diagnosis. 



### **Frontend**
Frontend folder contains the Streamlit web application code for EmoChat.
1. Home.py
   - main page for the Streamlit web application EmoChat.
     
2. pages/Emotion_Recognition_&_Depression_Detection.py
   - The code above defines a Streamlit page for emotion and depression symptom prediction using audio and text inputs.
     - `Audio Recording`: Captures and transcribes audio responses using a custom audio_recorder component.
     - `Emotion, Depression Prediction`: When all questions are answered, it sends the collected data to a Flask API (FLASK_API_URL) for emotion and depression analysis. The API returns predictions, which are stored in session state and displayed on the page.
     - `plot_symptom_heatmap()`: Visualizes depression symptoms using a heatmap.
     - `Session State`: Manages chat history, predicted emotions, and symptom data to ensure continuity in the chat.
   - Make sure to add your OpenAI API key in the `.env` file.
    
3. pages/Mental_Healthcare_Chatbot.py
   - Conversational interface for mental healthcare support, based on the user’s predicted emotions and depression symptoms.
   - Make sure to add your OpenAI API key in the `.env` file.
     

<br/>

## Work Flow
### 1. Environment Configuration
Before installing, create and activate a virtual conda environment with Python 3.8.19.
```
pip install flask torch=2.3.0 torchaudio urllib3 loralib accelerate
pip install streamlit==1.40.0 streamlit-float==0.3.5 streamlit-option-menu==0.3.2 audio_recorder_streamlit openai python-dotenv fastapi
```

### 2. Run Flask API server
```
cd Backend
python flask.py
```

### 3. Run Streamlit webpage
After starting the backend server, open the Streamlit page.
```
cd Frontend
streamlit run Home.py
```


