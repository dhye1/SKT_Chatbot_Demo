# Project demo
This repository includes the code and demo of our multimodal chatbot system.

## Folder Architecture
* **Backend**: Codes for
* **Frontend**: Codes for

All model checkpoints are separately stored in the [Dropbox][([https://www.dropbox.com/scl/fo/eij3dnkeccbyzuvc4qavm/AOs9R-huka-AIwXPsvd_0Ao?rlkey=tn5hqll9arnunbyz8odu3sz5y&st=59t5deol&dl=0)])
.

## Work Flow
### 1. Environment Configuration
Before installing, please create and activate a virtual conda environment with Python 3.8.19.
```
pip install flask torch=2.3.0 torchaudio urllib3 loralib accelerate
pip install streamlit==1.40.0 streamlit-float==0.3.5 streamlit-option-menu==0.3.2 audio_recorder_streamlit openai python-dotenv fastapi
```

### 2. Run Back-end
```
cd Backend
python flask.py
```

### 3. Run Front-end
After starting the backend server, open the Streamlit page.
```
cd Frontend
streamlit run Home.py
```


