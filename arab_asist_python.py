 import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# إعداد الصفحة
st.set_page_config(page_title="المساعد العربي الذكي", page_icon="🤖", layout="wide")
st.title("🤖 المساعد العربي الذكي")
st.write("اختر شخصية المساعد وتحدث معه بالعربية:")

# تحميل موديل عربي خفيف
model_name = "aubmindlab/aragpt2-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# الشخصيات المتاحة
personas = {
    "رسمية سياسية": "بأسلوب رسمي سياسي، أجب على السؤال التالي:",
    "تحليل علم نفس": "بأسلوب محلل نفسي، أجب على السؤال التالي:",
    "تعليم للأطفال": "بأسلوب مبسط يناسب الأطفال، أجب على السؤال التالي:",
    "شخصية ودودة": "بأسلوب ودود ومريح، أجب على السؤال التالي:"
}

# اختيار الشخصية
persona = st.radio("اختر شخصية المساعد:", list(personas.keys()))
question = st.text_area("اكتب سؤالك هنا:", placeholder="مثلاً: ما معنى الذكاء الاصطناعي؟")

# زر الإرسال
if st.button("إرسال"):
    prompt = f"{personas[persona]}\n{question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.success(answer)

# تحسين الشكل
st.markdown("""
<style>
body {background-color: #f5f7fa;}
textarea {border-radius: 10px;}
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
