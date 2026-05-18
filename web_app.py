import streamlit as st
from google import genai
from supabase import create_client
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Vibe Support AI", page_icon="💬")
st.title("🤖 Vibe Support")

# Cache clients to prevent recreation on every rerun
@st.cache_resource
def init_clients():
    return (
        genai.Client(api_key=st.secrets["GEMINI_KEY"]),
        create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]),
        InferenceClient(api_key=st.secrets["HF_TOKEN"])
    )

client, supabase, hf_client = init_clients()
	
def get_vector_fast(text):
    try:
        embedding = hf_client.feature_extraction(
            text,
            model="intfloat/multilingual-e5-large"
        )
        vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        return vector[0] if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list) else vector
    except Exception as e:
        st.error(f"Hugging Face Error: {e}")
        return None
                    
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
        vec = get_vector_fast(f"query: {prompt}")
        
        if vec:
            try:
                res = supabase.rpc("match_support", {
                    "query_embedding": vec, 
                    "match_threshold": 2.8, 
                    "match_count": 2
                }).execute()
                
                context_list = [f"المشكلة: {r['category']}\nالحل: {r['solution_agent']}" for r in res.data]
                context_text = "\n---\n".join(context_list) if context_list else "لا توجد معلومات مسترجعة مطابقة."
            except Exception as e:
                st.error(f"Supabase Error: {e}")
                context_text = "لا توجد معلومات مسترجعة مطابقة بسبب خطأ في قاعدة البيانات."
        else:
            context_text = "لا توجد معلومات مسترجعة مطابقة."

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

# تنسيق الإخراج
يجب أن يكون الجواب للموظف نصًا مباشرًا، يتبع اللهجة العراقية المهذبة والعملية، ويتضمن الحل أو الاستفسار المطلوب حسب السياق.
"""

        try:
            response = client.models.generate_content(
                model="models/gemini-2.5-flash", 
                contents=ai_prompt
            )
            output_text = response.text
            st.markdown(output_text)
            st.session_state.messages.append({"role": "assistant", "content": output_text})
        except Exception as e:
            st.error(f"حدث خطأ في استجابة الذكاء الاصطناعي: {e}")
