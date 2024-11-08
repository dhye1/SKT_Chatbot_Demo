# SKT AI Fellowship - Project demo
This repository includes the code and demo of our multimodal chatbot system.

## Folder Architecture
* **Backend**: Codes for
* **Frontend**: Codes for

All model checkpoints are separately stored in the [Dropbox](https://www.dropbox.com/home/skt_ai%20fellowship_model_checkpoints?di=left_nav_browse)
.

## Work Flow
### 1. Environment Configuration
Before installing, please create and activate a virtual conda environment with Python 3.8.19.
```
pip install -r requirements.txt
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


