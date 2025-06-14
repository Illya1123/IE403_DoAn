import demo as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, XLMRobertaTokenizer
import os
import joblib
import requests
import time
from urllib.parse import urlparse

def get_post_id(post_url):
    return urlparse(post_url).path.split('/')[-1]

def req(post_id, cursor):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': 'https://www.tiktok.com/',
        'Accept': 'application/json',
    }
    params = {
        'aid': '1988',
        'aweme_id': post_id,
        'cursor': str(cursor),
        'count': '20',
        'webcast_language': 'en',
    }
    try:
        response = requests.get(
            'https://www.tiktok.com/api/comment/list/',
            headers=headers,
            params=params,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def process_comments(data):
    if not data or 'comments' not in data:
        return [], False
    comments = []
    for cm in data['comments']:
        text = cm.get('text') or cm.get('share_info', {}).get('desc', '')
        if text:
            comments.append(text)
    return comments, data.get('has_more', 0) == 1

def crawl_comments_from_url(post_url, max_comments=50):
    post_id = get_post_id(post_url)
    comments = []
    cursor = 0
    has_more = True
    while has_more and len(comments) < max_comments:
        data = req(post_id, cursor)
        if not data:
            break
        batch, has_more = process_comments(data)
        comments.extend(batch)
        cursor += 20
        time.sleep(0.3)
    return comments[:max_comments]

# === Cấu hình chung ===
label_names = {
    0: 'Vui vẻ',
    1: 'Tức giận',
    2: 'Buồn bã',
    3: 'Sợ hãi',
    4: 'Trung lập'
}

# === Transformer models ===
MODEL_CONFIGS = {
    "PhoBERT-base": "./model/phobert-base/",
    "ViSoBERT": "./model/visobert/",
    "viBERT-base-cased": "./model/vibert-base-cased/",
    "RoBERTa-base-Vietnamese": "./model/roberta-base-vietnamese/"
}

@st.cache_resource
def load_transformer_models(model_dict):
    models = {}
    for name, model_dir in model_dict.items():
        try:
            if "visobert" in name.lower():
                tokenizer = XLMRobertaTokenizer.from_pretrained(
                    model_dir,
                    tokenizer_file=os.path.join(model_dir, "sentencepiece.bpe.model"),
                    use_fast=False
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

            model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=len(label_names))
            checkpoint_path = os.path.join(model_dir, "best_model.pth")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            model.to(device)
            model.eval()
            models[name] = (tokenizer, model, device)
        except Exception as e:
            st.error(f"❌ Lỗi khi load mô hình `{name}`: {e}")
    return models

def predict_transformer(text, tokenizer, model, device):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# === Machine Learning models ===
ML_MODEL_PATH = "./model/machine_learning/"
ML_MODEL_NAMES = [
    'Logistic Regression',
    'SVM',
    'Random Forest',
    'Stacking (LR + SVM + RF)'
]

@st.cache_resource
def load_ml_models(path, model_names):
    ml_models = {}
    vectorizer = joblib.load(os.path.join(path, "tfidf_vectorizer.pkl"))
    for name in model_names:
        try:
            model = joblib.load(os.path.join(path, f"{name}.pkl"))
            ml_models[name] = model
        except Exception as e:
            st.error(f"❌ Lỗi khi load ML model `{name}`: {e}")
    return vectorizer, ml_models

def predict_ml(text, model, vectorizer):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return label_names[pred]

# === Streamlit UI ===
st.set_page_config(page_title="So sánh cảm xúc từ nhiều mô hình", layout="centered")
st.title("So sánh dự đoán cảm xúc từ nhiều mô hình Tiếng Việt")

with st.spinner("🔄 Đang tải các mô hình Transformer..."):
    transformer_models = load_transformer_models(MODEL_CONFIGS)

with st.spinner("🔄 Đang tải các mô hình Machine Learning..."):
    vectorizer, ml_models = load_ml_models(ML_MODEL_PATH, ML_MODEL_NAMES)

st.success("✅ Tất cả mô hình đã sẵn sàng!")

user_input = st.text_area("✍️ Nhập bình luận TikTok:", height=150)

if st.button("📤 Dự đoán cảm xúc với tất cả mô hình"):
    if user_input.strip() == "":
        st.warning("⚠️ Vui lòng nhập một đoạn văn bản.")
    else:
        st.subheader("Kết quả từ mô hình Transformer:")
        for model_name, (tokenizer, model, device) in transformer_models.items():
            label_idx = predict_transformer(user_input, tokenizer, model, device)
            st.markdown(f"**{model_name}** → `{label_names[label_idx]}`")

        st.subheader("Kết quả từ mô hình Machine Learning:")
        for model_name, model in ml_models.items():
            label = predict_ml(user_input, model, vectorizer)
            st.markdown(f"**{model_name}** → `{label}`")

if st.checkbox("📋 Thử các ví dụ mẫu"):
    sample_texts = [
        "nghe bạn nam bảo : khổ thế nhờ đã dịch cô vít thì chớ mà xót🥺",
        "mày giỡn mặt tao à?",
        "trời ơi vui quá đi 😍 hôm nay được nghỉ học",
        "bạn nữ đó nhìn buồn thật sự",
        "tôi cảm thấy lo lắng khi ra đường bây giờ"
    ]
    for text in sample_texts:
        st.markdown(f"---\n**📝 Bình luận:** {text}")
        for model_name, (tokenizer, model, device) in transformer_models.items():
            pred = label_names[predict_transformer(text, tokenizer, model, device)]
            st.markdown(f"**{model_name}** → `{pred}`")
        for model_name, model in ml_models.items():
            pred = predict_ml(text, model, vectorizer)
            st.markdown(f"**{model_name}** → `{pred}`")


st.markdown("---")
st.header("Dự đoán cảm xúc từ bình luận TikTok")

tiktok_url = st.text_input("🔗 Dán link video TikTok tại đây:")
max_cmt = st.slider("🔢 Số lượng bình luận muốn phân tích", 1, 1000, 10)

if st.button("📥 Crawl & Dự đoán bình luận TikTok"):
    if not tiktok_url.strip():
        st.warning("⚠️ Vui lòng nhập link TikTok.")
    else:
        with st.spinner("🔄 Đang crawl dữ liệu từ TikTok..."):
            comments = crawl_comments_from_url(tiktok_url, max_comments=max_cmt)

        if not comments:
            st.error("🚫 Không lấy được bình luận hoặc link không hợp lệ.")
        else:
            st.success(f"✅ Lấy được {len(comments)} bình luận. Đang phân tích...")

            for idx, text in enumerate(comments):
                st.markdown(f"---\n**📝 Bình luận #{idx+1}:** {text}")
                
                for model_name, (tokenizer, model, device) in transformer_models.items():
                    pred = label_names[predict_transformer(text, tokenizer, model, device)]
                    st.markdown(f"**{model_name}** → `{pred}`")
                
                for model_name, model in ml_models.items():
                    pred = predict_ml(text, model, vectorizer)
                    st.markdown(f"**{model_name}** → `{pred}`")