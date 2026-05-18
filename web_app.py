import streamlit as st
from google import genai
from supabase import create_client
import requests
import time

st.set_page_config(page_title="Vibe Support AI", page_icon="💬")
st.title("🤖 Vibe Support")

client = genai.Client(api_key=st.secrets["GEMINI_KEY"])
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
HF_TOKEN = st.secrets["HF_TOKEN"]

    
def get_vector_fast(text):
    api_url = "https://huggingface.co"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
        
        # Try up to 3 times if the model is still loading
    for _ in range(3):
        response = requests.post(api_url, json={"inputs": text}, headers=headers)
        result = response.json()
            
            # If HF says "Model is loading", wait 5 seconds and try again
        if isinstance(result, dict) and "estimated_time" in result:
            time.sleep(5)
            continue
                
            # If it's a valid list (the vector), return it
        if isinstance(result, list):
                # The model returns [[vector]] for this specific API, so we take the first element
            return result[0] if isinstance(result[0], list) else result
                
        # If all fails
    st.error("Hugging Face API is busy. Try again in a moment.")
    return None
    
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
