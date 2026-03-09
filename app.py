import streamlit as st
import pandas as pd
import numpy as np
import difflib
import pickle
import os
import re
from huggingface_hub import InferenceClient

st.set_page_config(page_title="GrammarAI", page_icon="✦", layout="centered")

st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown("""
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
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_data():
    base = os.path.dirname(os.path.abspath(__file__))

    # Load Grammar Correction CSV
    grammar_path = os.path.join(base, 'Grammar Correction.csv')
    if not os.path.exists(grammar_path):
        grammar_path = os.path.join(base, 'Grammar_Correction.csv')
    
    df_grammar = pd.read_csv(grammar_path)
    df_grammar['Ungrammatical Statement'] = df_grammar['Ungrammatical Statement'].apply(
        lambda x: re.sub(r'\s+', ' ', re.sub(r'^\d+\.\s*', '', str(x).strip())) if isinstance(x, str) else '')
    df_grammar['Standard English'] = df_grammar['Standard English'].apply(
        lambda x: re.sub(r'\s+', ' ', re.sub(r'^\d+\.\s*', '', str(x).strip())) if isinstance(x, str) else '')
    df_grammar['wrong_lower'] = df_grammar['Ungrammatical Statement'].str.lower()
    df_grammar['correct_lower'] = df_grammar['Standard English'].str.lower()
    df_grammar = df_grammar.drop_duplicates(subset=['wrong_lower']).reset_index(drop=True)

    # Load vocab & guides
    vocab_path = os.path.join(base, 'vocabulary_idioms.csv')
    phrasal_path = os.path.join(base, 'phrasal_verbs.csv')
    expanded_path = os.path.join(base, 'vocabulary_expanded.csv')
    guide_complete_path = os.path.join(base, 'english_complete_guide.txt')
    guide_toefl_path = os.path.join(base, 'english_guide.txt')

    guide_complete, guide_toefl = '', ''
    if os.path.exists(guide_complete_path):
        with open(guide_complete_path, 'r', encoding='utf-8') as f:
            guide_complete = f.read()
    if os.path.exists(guide_toefl_path):
        with open(guide_toefl_path, 'r', encoding='utf-8') as f:
            guide_toefl = f.read()

    return {
        'grammar': df_grammar,
        'guide_complete': guide_complete,
        'guide_toefl': guide_toefl
    }

try:
    data = load_all_data()
    df = data['grammar']
    guide = data['guide_complete'] + '\n' + data['guide_toefl']
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

SYSTEM_PROMPT = f"""
Kamu adalah GrammarAI — asisten bahasa Inggris yang cerdas, friendly, dan sangat membantu untuk pelajar Indonesia.

Knowledge base kamu:
{guide[:3000]}

Kamu bisa:
1. Koreksi grammar — tunjukkan kesalahan + penjelasan lengkap
2. Translate Indonesia ke English dengan natural
3. Jelaskan grammar rules dengan bahasa mudah
4. Kasih contoh kalimat yang benar
5. Jawab tentang vocabulary, idioms, phrasal verbs, TOEFL/IELTS

Format jawaban:
- Penjelasan pakai Bahasa Indonesia
- Contoh kalimat dalam English
- Untuk koreksi: ❌ salah → ✅ benar, lalu 📌 penjelasan
- Friendly dan encouraging!
"""

if 'messages' not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.markdown("""
    <div style='display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:1.5rem;'>
        <div style='background:#ede9ff;border:1px solid #7c6bff;padding:8px 16px;border-radius:20px;font-size:0.85rem;color:#5a3fcc;'>✏️ Koreksi kalimat saya</div>
        <div style='background:#ffe0ef;border:1px solid #ff6b9d;padding:8px 16px;border-radius:20px;font-size:0.85rem;color:#c2185b;'>🌐 Translate ke English</div>
        <div style='background:#e0f5ee;border:1px solid #06d6a0;padding:8px 16px;border-radius:20px;font-size:0.85rem;color:#087f5b;'>❓ Tanya grammar</div>
        <div style='background:#fff9e0;border:1px solid #ffd43b;padding:8px 16px;border-radius:20px;font-size:0.85rem;color:#856f00;'>📚 Tips TOEFL/IELTS</div>
    </div>
    """, unsafe_allow_html=True)

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
                quick_reply = f"""
<div style='background:#f8f4ff;border-left:4px solid #7c6bff;border-radius:12px;padding:1.2rem;margin:4px 0;'>
  <div style='font-size:1rem;line-height:2;'>{diff_html}</div>
  <div style='margin-top:10px;color:#087f5b;font-weight:700;'>✓ {grammar_result['corrected']}</div>
  <div style='margin-top:8px;'>
    <span style='background:#ede9ff;color:#5a3fcc;padding:3px 10px;border-radius:20px;font-size:0.78rem;'>{grammar_result['error_type']}</span>
    <span style='color:#999;font-size:0.75rem;margin-left:8px;'>confidence: {conf}%</span>
  </div>
</div>"""
                ai_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Kalimat '{prompt}' dikoreksi menjadi '{grammar_result['corrected']}'. Error: {grammar_result['error_type']}. Jelaskan kenapa salah dan apa aturannya. Maksimal 3 kalimat, pakai bahasa Indonesia."}
                ]
                explanation = ask_hf(ai_messages)
                full_reply = quick_reply + f"\n\n📌 **Penjelasan:** {explanation}"
            else:
                ai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for m in st.session_state.messages:
                    ai_messages.append({"role": m['role'], "content": m['content']})
                full_reply = ask_hf(ai_messages)

            st.markdown(full_reply, unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', 'content': full_reply})

if st.session_state.messages:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
