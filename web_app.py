import streamlit as st
from google import genai
from supabase import create_client
import requests

st.set_page_config(page_title="Vibe Support AI", page_icon="💬")
st.title("🤖 Vibe Support")

client = genai.Client(api_key=st.secrets["GEMINI_KEY"])
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
HF_TOKEN = st.secrets["HF_TOKEN"]

# 3. Function to get Vector without loading the model locally
def get_vector_fast(text):
    api_url = "https://huggingface.co"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(api_url, json={"inputs": text}, headers=headers)
    return response.json()

# 4. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("شلون أگدر أساعدك خالي؟"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # A. Search
        vec = get_vector_fast(f"query: {prompt}")
        res = supabase.rpc("match_support", {"query_embedding": vec, "match_threshold": 0.4, "match_count": 2}).execute()
        
        # B. Generate Answer
        context = "\n".join([f"المشكلة: {r['category']}\nالحل: {r['solution_agent']}" for r in res.data])
        ai_prompt = f"أجب بلهجة عراقية بناءً على: {context}\nالسؤال: {prompt}"
        
        full_response = client.models.generate_content(model="models/gemini-1.5-flash", contents=ai_prompt).text
        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
