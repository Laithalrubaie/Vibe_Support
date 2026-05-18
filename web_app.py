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
# B. Generate Answer
# نجمع كل النتائج اللي لقاها السيرش ونرتبها للـ AI
context_list = []
for r in res.data:
    context_list.append(f"المشكلة: {r['category']}\nالحل: {r['solution_agent']}")

# إذا ماكو نتائج من السيرش، ننطي نص فارغ حتى الـ AI يگول "ما عندي معلومة"
context_text = "\n---\n".join(context_list) if context_list else "لا توجد معلومات مسترجعة مطابقة."

ai_prompt = f"""
أنت مساعد خبير لموظفي الدعم الفني في منصة تعليمية عراقية. مهمتك هي إعطاء حلول دقيقة بناءً على "المعلومات المسترجعة" فقط.

# قواعد العمل
1. **التعامل مع الأسئلة العامة أو غير الواضحة:** إذا كان سؤال الموظف عامًا أو غير واضح (مثال: "الطالب عنده مشكلة")، لا تقدم حلاً مباشرًا. بدلاً من ذلك، اطلب من الموظف تقديم تفاصيل إضافية (مثل: "ما هو نوع المشكلة؟"، "في أي صف؟"، "ما هي الرسالة التي تظهر للطالب؟").
2. **التعامل مع المعلومات غير المتوفرة:** إذا لم تتضمن "المعلومات المسترجعة" حلاً مباشرًا للسؤال المطروح، أجب بـ "هذه المعلومة غير متوفرة عندي حالياً، يرجى سؤال المشرف". لا تحاول تأليف حل من عندك.
3. **اللهجة والأسلوب:** التزم بلهجة عراقية مهذبة وعملية في جميع ردودك.
4. **نقص معلومات الطالب:** إذا كانت هناك حاجة لمعلومات إضافية عن الطالب (مثل كود الطالب أو بريده الإلكتروني) لإيجاد حل، ذكّر الموظف بضرورة طلب هذه المعلومات من الطالب أولاً قبل المتابعة.

# المعلومات المسترجعة
المعلومات المتوفرة لديك هي:
{context_text}

# سؤال الموظف
سؤال الموظف الذي تحتاج إلى الإجابة عليه هو: {prompt}

# الخطوات
1. اقرأ سؤال الموظف بعناية.
2. راجع "المعلومات المسترجعة" لتحديد ما إذا كان هناك حل مباشر أو معلومات ذات صلة بسؤال الموظف.
3. إذا كان السؤال عامًا أو غير واضح، اطلب مزيدًا من التفاصيل من الموظف.
4. إذا كانت "المعلومات المسترجعة" لا تحتوي على حل، أبلغ الموظف بعدم توفر المعلومة.
5. إذا كانت هناك حاجة لمعلومات إضافية عن الطالب، اطلب من الموظف الحصول عليها أولاً.
6. إذا وجدت حلاً مباشرًا في "المعلومات المسترجعة"، قدمه للموظف بلهجة عراقية مهذبة وعملية.

# تنسيق الإخراج
يجب أن يكون الجواب للموظف نصًا مباشرًا، يتبع اللهجة العراقية المهذبة والعملية، ويتضمن الحل أو الاستفسار المطلوب حسب السياق.
"""

try:
    response = client.models.generate_content(
        model="models/gemini-2.5-flash", 
        contents=ai_prompt
    )
    st.markdown(response.text)
    st.session_state.messages.append({"role": "assistant", "content": response.text})
except Exception as e:
    st.error(f"حدث خطأ في استجابة الذكاء الاصطناعي: {e}")
    full_response = client.models.generate_content(model="models/gemini-2.5-flash", contents=ai_prompt).text
    st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
