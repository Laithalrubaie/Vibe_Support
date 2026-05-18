import streamlit as st
from google import genai
from supabase import create_client
import requests
import time
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Vibe Support AI", page_icon="💬")
st.title("🤖 Vibe Support")

client = genai.Client(api_key=st.secrets["GEMINI_KEY"])
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
hf_client = InferenceClient(api_key=st.secrets["HF_TOKEN"])
    
def get_vector_fast(text):
            try:
                # Use the client to get the embedding
                embedding = hf_client.feature_extraction(
                    text,
                    model="intfloat/multilingual-e5-large"
                )
                
                # Convert to list if it's not already
                if hasattr(embedding, "tolist"):
                    vector = embedding.tolist()
                else:
                    vector = list(embedding)
        
                # The model E5 returns [[vector]] sometimes, we need [vector]
                if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
                    return vector[0]
                return vector
                
            except Exception as e:
                st.error(f"Hugging Face Error: {e}")
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
