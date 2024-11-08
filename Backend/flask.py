from flask import Flask, request, jsonify
import torch
import torchaudio
from transformers import RobertaTokenizer
from safetensors.torch import load_file
from prediction import TextAudioClassifier as EmotionClassifier
import depression_prediction
from custom_roberta import RobertaCrossAttn
from wavlm_plus import WavLMWrapper
from whisper_model import WhisperWrapper
from argparse import Namespace
import os

app = Flask(__name__)

# GPU/CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 감정 예측 모델의 args
emotion_args = Namespace(
    audio_model="whisper-medium",
    text_model="roberta-large",
    speaker="wavlm",
    hidden_dim=256,
    dr=0.1,
    cross_modal_atten=False,
    modal="multimodal",
    downstream=False,
    finetune_method="adapter",
    adapter_hidden_dim=128,
    lora_rank=16,
    is_key_lora=False,
    wg=False,
    use_conv_output=False,
    ws=False,
    max_txt_len=256,
    self_attn=True,
    num_hidden_layers=12,
    embedding_prompt_dim=5,
    finetune_roberta=False
)

# 우울증 예측 모델의 args
depression_args = Namespace(
    audio_model="whisper-medium",
    text_model="roberta-large",
    speaker="wavlm",
    hidden_dim=256,
    dr=0.1,
    cross_modal_atten=False,
    modal="multimodal",
    downstream=False,
    phq_mode="depression",
    finetune_method="lora",
    wg=False,
    use_conv_output=False,
    ws=False,
    lora_rank=16,
    max_txt_len=256,
    num_hidden_layers=12,
    embedding_prompt_dim=5,
    self_attn=True,
    lora_alpha=16,
    lora_dropout=0.1,
    lora_target_modules=("key", "query", "value"),
    adapter_hidden_dim=128,
    is_key_lora=False,
    inference_mode=True,
    finetune_roberta=True
)

# 감정 예측 모델 설정 및 가중치 로드 함수
def load_emotion_model():
    try:
        print("Loading emotion model...")
        audio_model = WhisperWrapper(emotion_args).to(device)
        text_model = RobertaCrossAttn(emotion_args, audio_model).to(device)
        speaker_model = WavLMWrapper(emotion_args).to(device)
        
        emotion_model = EmotionClassifier(
            audio_model=audio_model,
            text_model=text_model,
            speaker_model=speaker_model,
            speaker=emotion_args.speaker,
            audio_dim=1024,
            text_dim=1024,
            speaker_dim=768,
            hidden_dim=emotion_args.hidden_dim,
            num_classes=6,
            dropout_prob=emotion_args.dr,
            cross_modal_atten=emotion_args.cross_modal_atten,
            modal=emotion_args.modal
        ).to(device)

        # 가중치 로드
        audio_model_path = '/home/user/emo/demo/inference/1024_merge_crossattn_whisper_merged_fold_1.pt'
        pred_model_path = '/home/user/emo/demo/inference/1024_merge_crossattn_pred_fold_1.pt'
        speaker_state_dict_path = '/home/user/emo/demo/inference/1024_merge_crossattn_whisper_fold_1.pt'
        text_model_path = '/home/user/emo/demo/inference/model.safetensors'

        pred_state_dict = torch.load(pred_model_path, map_location=device)
        speaker_state_dict = torch.load(speaker_state_dict_path, map_location=device)
        audio_state_dict = torch.load(audio_model_path, map_location=device)
        text_state_dict = load_file(text_model_path)

        if emotion_model.speaker_model is not None:
            emotion_model.speaker_model.load_state_dict(speaker_state_dict, strict=False)
        emotion_model.pred_linear.load_state_dict(pred_state_dict, strict=False)
        emotion_model.text_model.semantic_model.load_state_dict(text_state_dict, strict=False)
        emotion_model.audio_model.backbone_model.load_state_dict(audio_state_dict, strict=False)

        emotion_model.eval()
        print("Emotion model loaded successfully.")
        return emotion_model
    except Exception as e:
        print(f"Emotion model loading error: {e}")
        return None

# 우울증 예측 모델 설정 및 가중치 로드 함수
def load_depression_model():
    try:
        print("Loading depression model...")
        audio_model = WhisperWrapper(depression_args).to(device)
        text_model = RobertaCrossAttn(depression_args, audio_model).to(device)
        speaker_model = WavLMWrapper(depression_args).to(device)
        
        depression_model = depression_prediction.TextAudioClassifier(
            audio_model=audio_model,
            text_model=text_model,
            speaker_model=speaker_model,
            speaker=depression_args.speaker,
            audio_dim=1024,
            text_dim=1024,
            speaker_dim=768,
            hidden_dim=depression_args.hidden_dim,
            num_classes=2,
            dropout_prob=depression_args.dr,
            cross_modal_atten=depression_args.cross_modal_atten,
            modal=depression_args.modal,
            phq_mode="depression",
            num_symptom=5
        ).to(device)

        # 가중치 로드
        audio_model_path = '/home/user/emo/demo/inference/dep_pt/dep_1024_crossattn(False)_ffn_(kqv)_whisper_merged_fold_1.pt'
        speaker_state_dict = '/home/user/emo/demo/inference/dep_pt/dep_1024_crossattn(False)_ffn_(kqv)_whisper_fold_1.pt'
        text_model_path = '/home/user/emo/demo/inference/dep_pt/model.safetensors'
        sym_model_path = '/home/user/emo/demo/inference/dep_pt/dep_1024_crossattn(False)_ffn_(kqv)_pred_fold_1.pt'
        dpr_model_path = '/home/user/emo/demo/inference/dep_pt/dep_1024_crossattn(False)_ffn_(kqv)_classifier_fold_1.pt'
        dpr_wd_path = '/home/user/emo/demo/inference/dep_pt/dep_1024_crossattn(False)_ffn_(kqv)_wd_fold_1.pt'
        dpr_bd_path = '/home/user/emo/demo/inference/dep_pt/dep_1024_crossattn(False)_ffn_(kqv)_bd_fold_1.pt'

        audio_state_dict = torch.load(audio_model_path, map_location=device)
        speaker_state_dict = torch.load(speaker_state_dict, map_location=device)
        text_state_dict = load_file(text_model_path)
        sym_state_dict = torch.load(sym_model_path)
        dpr_state_dict = torch.load(dpr_model_path)
        dpr_wd_dict = torch.load(dpr_wd_path)
        dpr_bd_dict = torch.load(dpr_bd_path)

        if depression_model.speaker_model is not None:
            depression_model.speaker_model.load_state_dict(speaker_state_dict, strict=False)
        if depression_model.audio_model is not None:
            depression_model.audio_model.backbone_model.load_state_dict(audio_state_dict, strict=False)
        if depression_model.text_model is not None:
            if "embeddings.position_ids" in text_state_dict:
                del text_state_dict["embeddings.position_ids"]
            depression_model.text_model.semantic_model.load_state_dict(text_state_dict, strict=True)

        depression_model.phq8_classifiers.load_state_dict(sym_state_dict, strict=False)
        depression_model.phq8_binary_classifier.load_state_dict(dpr_state_dict, strict=False)
        depression_model.W_d.load_state_dict(dpr_wd_dict, strict=True)
        depression_model.b_d = dpr_bd_dict

        depression_model.eval()
        print("Depression model loaded successfully.")
        return depression_model
    except Exception as e:
        print(f"Depression model loading error: {e}")
        return None

# 감정 예측 함수
def predict_emotion(audio_file_path, text):
    print("Starting emotion prediction...")
    audio_waveform, _ = torchaudio.load(audio_file_path)
    audio_waveform = audio_waveform[0]
    print("Loaded audio waveform:", audio_waveform.shape)

    max_length_in_seconds = 30
    sample_rate = 16000
    max_length = max_length_in_seconds * sample_rate

    if audio_waveform.size(0) > max_length:
        audio_waveform = audio_waveform[:max_length]
    else:
        padding_size = max_length - audio_waveform.size(0)
        audio_waveform = torch.nn.functional.pad(audio_waveform, (0, padding_size))
    print("Audio waveform after padding:", audio_waveform.shape)

    text_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)['input_ids']
    print("Tokenized text:", text_tokens)

    audio_waveform = audio_waveform.to(device)
    text_tokens = text_tokens.to(device)

    with torch.no_grad():
        outputs = emotion_model(audio_input=audio_waveform.unsqueeze(0), text_input=text_tokens)
    print("Model outputs:", outputs)

    predicted_label = torch.argmax(outputs, axis=1).item()
    print("Predicted label:", predicted_label)

    return emotion_label_dict[predicted_label]

# 우울증 예측 함수
def predict_depression_symptoms(audio_file_path, text):
    print("Starting depression prediction...")
    audio_waveform, _ = torchaudio.load(audio_file_path)
    audio_waveform = audio_waveform[0]
    print("Loaded audio waveform:", audio_waveform.shape)

    max_length_in_seconds = 30
    sample_rate = 16000
    max_length = max_length_in_seconds * sample_rate

    if audio_waveform.size(0) > max_length:
        audio_waveform = audio_waveform[:max_length]
    else:
        padding_size = max_length - audio_waveform.size(0)
        audio_waveform = torch.nn.functional.pad(audio_waveform, (0, padding_size))
    print("Audio waveform after padding:", audio_waveform.shape)

    text_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)['input_ids']
    print("Tokenized text:", text_tokens)

    audio_waveform = audio_waveform.to(device)
    text_tokens = text_tokens.to(device)

    with torch.no_grad():
        outputs = depression_model(audio_input=audio_waveform.unsqueeze(0), text_input=text_tokens)
    print("Model outputs:", outputs)

    hidden_outputs, depression_output = outputs
    print("Hidden outputs:", hidden_outputs)
    print("Depression output:", depression_output)

    mean_symptom_outputs = hidden_outputs.mean(dim=-1)
    weights = torch.softmax(depression_model.W_d(mean_symptom_outputs) + depression_model.b_d, dim=1).squeeze()
    print("Symptom weights:", weights)

    symptom_weights = weights.detach().cpu().numpy().astype(float)  # Convert to Python float
    depression_label_index = int((depression_output > 0.5).item())
    depression_prediction = depression_label_dict["depression"].get(str(depression_label_index), "unknown")
    print("Predicted depression:", depression_prediction)

    symptoms_with_weights = {
        depression_label_dict["symptom"][str(i)]: round(symptom_weights[i], 4)
        for i in range(len(symptom_weights))
    }
    print("Symptoms with weights:", symptoms_with_weights)

    return depression_prediction, symptoms_with_weights


# 모델 및 토크나이저 초기화
emotion_model = load_emotion_model()
depression_model = load_depression_model()
tokenizer = RobertaTokenizer.from_pretrained(emotion_args.text_model)

# Label 매핑
emotion_label_dict = {0: "neutral", 1: "sadness", 2: "frustrated", 3: "anger", 4: "happy", 5: "excited"}
depression_label_dict = {
    "depression": {"0": "non", "1": "dep"},
    "symptom": {
        "0": "phq_nointerest", "1": "phq_depressed", "2": "phq_sleep",
        "3": "phq_tired", "4": "phq_failure", "5": "phq_appetite",
        "6": "phq_concentrating", "7": "phq_moving"
    }
}

# 예측 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    if emotion_model is None or depression_model is None:
        print("Model loading failed.")
        return jsonify({"error": "Model loading failed"}), 500
    
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio_file']
    text = request.form.get('text', '')

    # 오디오 파일 저장
    audio_file_path = "received_audio.wav"
    audio_file.save(audio_file_path)

    try:
        predicted_emotion = predict_emotion(audio_file_path, text)
        depression_prediction, symptom_weights = predict_depression_symptoms(audio_file_path, text)

        os.remove(audio_file_path)  # 처리 완료 후 파일 삭제
        
        return jsonify({
            'emotion_prediction': predicted_emotion,
            'depression_prediction': depression_prediction,
            'symptom_weights': symptom_weights
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
