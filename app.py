# CELL 6 — Tulis app.py dengan fitur TTS + Upload PDF/Gambar
PROJECT_DIR = '/content/drive/MyDrive/grammar-chatbot'

app_code = """
import streamlit as st
import pandas as pd
import numpy as np
import difflib
import pickle
import os
import re
import base64
from huggingface_hub import InferenceClient
from gtts import gTTS
import tempfile

st.set_page_config(page_title="GrammarAI", page_icon="✦", layout="centered")

st.markdown(\"\"\"
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono&family=Lato:wght@300;400;700&display=swap');
html, body, .stApp {
    background-color: #f0f4ff !important;
    color: #1a1a2e !important;
    font-family: 'Lato', sans-serif !important;
}
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stChatInput"] textarea {
    background: white !important;
    color: #1a1a2e !important;
    border: 2px solid #7c6bff !important;
    border-radius: 14px !important;
}
.stButton button {
    background: linear-gradient(135deg, #7c6bff, #5a4fcf) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}
[data-testid="stChatMessage"] {
    background: white !important;
    border-radius: 16px !important;
    padding: 12px !important;
    margin: 8px 0 !important;
    box-shadow: 0 2px 8px rgba(124,107,255,0.1) !important;
}
</style>
\"\"\", unsafe_allow_html=True)

st.markdown(\"\"\"
<div style='text-align:center;padding:2rem 0 1rem;'>
  <div style='font-family:Syne;font-size:2.5rem;font-weight:800;
              background:linear-gradient(135deg,#7c6bff,#ff6b9d);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    ✦ GrammarAI Chat
  </div>
  <div style='color:#666;font-size:0.9rem;margin-top:6px;'>
    Tanya apapun tentang bahasa Inggris · Koreksi · Translate · Explain
  </div>
</div>
\"\"\", unsafe_allow_html=True)

@st.cache_resource
def load_all_data():
    base = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base, 'models')
    with open(os.path.join(model_dir, 'all_data.pkl'), 'rb') as f:
        return pickle.load(f)

try:
    data = load_all_data()
    df = data['grammar']
    guide = data['guide_complete'] + '\\\\n' + data['guide_toefl']
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

HF_TOKEN = os.environ.get('HF_TOKEN', '')

def ask_hf(messages):
    models_to_try = [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "google/gemma-2-2b-it",
        "Qwen/Qwen2.5-72B-Instruct",
        "HuggingFaceH4/zephyr-7b-beta"
    ]
    client = InferenceClient(token=HF_TOKEN)
    for model in models_to_try:
        try:
            response = client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=600,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception:
            continue
    return "Maaf, semua model AI sedang tidak tersedia. Coba lagi nanti ya!"

def text_to_speech(text):
    try:
        clean = re.sub(r'<[^>]+>', '', text)
        clean = re.sub(r'[*#📌✓❌✅↗]', '', clean).strip()
        tts = gTTS(text=clean[:500], lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            tts.save(f.name)
            with open(f.name, 'rb') as audio:
                audio_bytes = audio.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        return f'<audio controls autoplay style="width:100%;margin-top:8px;"><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
    except:
        return ''

def extract_text_from_file(uploaded_file):
    text = ''
    if uploaded_file.type == 'application/pdf':
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() + '\\\\n'
        except:
            text = 'Gagal membaca PDF.'
    elif uploaded_file.type.startswith('image'):
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(uploaded_file)
            text = pytesseract.image_to_string(img)
        except:
            text = 'Gagal membaca gambar.'
    return text.strip()

def exact_match(text):
    low = text.strip().lower()
    m = df[df['wrong_lower'] == low]
    if len(m):
        r = m.iloc[0]
        return {'corrected': r['Standard English'], 'error_type': r['Error Type'], 'confidence': 1.0}
    return None

def fuzzy_match(text, threshold=0.82):
    low = text.strip().lower()
    best, idx = 0, -1
    for i, w in enumerate(df['wrong_lower']):
        s = difflib.SequenceMatcher(None, low, w).ratio()
        if s > best: best, idx = s, i
    if best >= threshold and idx >= 0:
        r = df.iloc[idx]
        return {'corrected': r['Standard English'], 'error_type': r['Error Type'], 'confidence': round(best, 2)}
    return None

def format_diff(wrong, correct):
    ww, cw = wrong.split(), correct.split()
    tokens = []
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(None,ww,cw).get_opcodes():
        if tag=='equal':
            for w in ww[i1:i2]: tokens.append({'word':w,'tag':'same'})
        elif tag=='replace':
            for w in ww[i1:i2]: tokens.append({'word':w,'tag':'removed'})
            for w in cw[j1:j2]: tokens.append({'word':w,'tag':'added'})
        elif tag=='delete':
            for w in ww[i1:i2]: tokens.append({'word':w,'tag':'removed'})
        elif tag=='insert':
            for w in cw[j1:j2]: tokens.append({'word':w,'tag':'added'})
    html = ''
    for t in tokens:
        if t['tag']=='removed':
            html += f'<span style="color:#ff4d6d;text-decoration:line-through;background:#ffe0e6;padding:2px 6px;border-radius:4px;margin:0 2px;">{t["word"]}</span> '
        elif t['tag']=='added':
            html += f'<span style="color:#087f5b;background:#d3f9d8;padding:2px 6px;border-radius:4px;margin:0 2px;">↗ {t["word"]}</span> '
        else:
            html += f'{t["word"]} '
    return html

SYSTEM_PROMPT = f\"\"\"
Kamu adalah GrammarAI — asisten bahasa Inggris yang cerdas, friendly, dan sangat membantu untuk pelajar Indonesia.
Knowledge base: {guide[:3000]}
Kamu bisa koreksi grammar, translate, jelaskan rules, kasih contoh, jawab tentang vocab/idioms/TOEFL/IELTS.
Format: penjelasan pakai Bahasa Indonesia, contoh dalam English, friendly dan encouraging!
\"\"\"

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'tts_enabled' not in st.session_state:
    st.session_state.tts_enabled = False

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    st.session_state.tts_enabled = st.toggle("🔊 Text to Speech", value=st.session_state.tts_enabled)
    if st.session_state.tts_enabled:
        st.success("Jawaban akan diputar otomatis!")

    st.markdown("---")
    st.markdown("### 📄 Upload File")
    uploaded_file = st.file_uploader("Upload PDF atau Gambar", type=['pdf', 'png', 'jpg', 'jpeg'])
    if uploaded_file:
        with st.spinner("Membaca file..."):
            extracted = extract_text_from_file(uploaded_file)
            if extracted:
                st.success(f"✅ Teks berhasil dibaca! ({len(extracted)} karakter)")
                if st.button("🔍 Analisis teks ini"):
                    prompt_file = f"Tolong analisis dan koreksi grammar dari teks berikut:\\\\n\\\\n{extracted[:1000]}"
                    st.session_state.messages.append({'role': 'user', 'content': f'📄 Analisis file: {uploaded_file.name}'})
                    ai_messages = [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": prompt_file}]
                    reply = ask_hf(ai_messages)
                    if st.session_state.tts_enabled:
                        reply += text_to_speech(reply)
                    st.session_state.messages.append({'role': 'assistant', 'content': reply})
                    st.rerun()

if len(st.session_state.messages) == 0:
    st.markdown(\"\"\"
    <div style='display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:1.5rem;'>
        <div style='background:#ede9ff;border:1px solid #7c6bff;padding:8px 16px;border-radius:20px;font-size:0.85rem;color:#5a3fcc;'>✏️ Koreksi kalimat saya</div>
        <div style='background:#ffe0ef;border:1px solid #ff6b9d;padding:8px 16px;border-radius:20px;font-size:0.85rem;color:#c2185b;'>🌐 Translate ke English</div>
        <div style='background:#e0f5ee;border:1px solid #06d6a0;padding:8px 16px;border-radius:20px;font-size:0.85rem;color:#087f5b;'>❓ Tanya grammar</div>
        <div style='background:#fff9e0;border:1px solid #ffd43b;padding:8px 16px;border-radius:20px;font-size:0.85rem;color:#856f00;'>📚 Tips TOEFL/IELTS</div>
    </div>
    \"\"\", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'], unsafe_allow_html=True)

if prompt := st.chat_input("Ketik kalimat untuk dikoreksi, atau tanya apapun..."):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            grammar_result = exact_match(prompt) or fuzzy_match(prompt, 0.82) or fuzzy_match(prompt, 0.65)

            if grammar_result:
                diff_html = format_diff(prompt, grammar_result['corrected'])
                conf = int(grammar_result['confidence'] * 100)
                quick_reply = f\"\"\"
<div style='background:#f8f4ff;border-left:4px solid #7c6bff;border-radius:12px;padding:1.2rem;margin:4px 0;'>
  <div style='font-size:1rem;line-height:2;'>{diff_html}</div>
  <div style='margin-top:10px;color:#087f5b;font-weight:700;'>✓ {grammar_result['corrected']}</div>
  <div style='margin-top:8px;'>
    <span style='background:#ede9ff;color:#5a3fcc;padding:3px 10px;border-radius:20px;font-size:0.78rem;'>{grammar_result['error_type']}</span>
    <span style='color:#999;font-size:0.75rem;margin-left:8px;'>confidence: {conf}%</span>
  </div>
</div>\"\"\"
                ai_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Kalimat '{prompt}' dikoreksi menjadi '{grammar_result['corrected']}'. Error: {grammar_result['error_type']}. Jelaskan kenapa salah dan apa aturannya. Maksimal 3 kalimat, pakai bahasa Indonesia."}
                ]
                explanation = ask_hf(ai_messages)
                full_reply = quick_reply + f"\\\\n\\\\n📌 **Penjelasan:** {explanation}"
            else:
                ai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for m in st.session_state.messages:
                    ai_messages.append({"role": m['role'], "content": m['content']})
                full_reply = ask_hf(ai_messages)

            if st.session_state.tts_enabled:
                full_reply += text_to_speech(full_reply)

            st.markdown(full_reply, unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', 'content': full_reply})

if st.session_state.messages:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
"""

with open(f'{PROJECT_DIR}/app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

print('✅ app.py berhasil diupdate dengan fitur TTS + Upload PDF/Gambar!')
print(f'📁 Lokasi: {PROJECT_DIR}/app.py')
