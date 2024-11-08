# SKT AI Fellowship - project demo
This repository includes the code and demo of our multimodal chatbot system.

## Folder Architecture
* **Backend**: Codes for
* **Frontend**: Codes for

All model checkpoints are separately stored in the [Dropbox](https://www.dropbox.com/scl/fo/y9ydmvv0bj846klkfdin0/h?rlkey=epyzclz2kbcf2g4iuz0tojlm9&dl=0)


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


