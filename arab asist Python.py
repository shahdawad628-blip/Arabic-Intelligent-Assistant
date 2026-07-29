import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# تحميل موديل خفيف مناسب لـ Streamlit Cloud
model_name = "aubmindlab/aragpt2-medium"   # ممكن تستبدليه بـ "tiiuae/falcon-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# إعداد واجهة Streamlit
st.set_page_config(page_title="المساعد العربي الذكي", page_icon="🤖", layout="wide")
st.title("🤖 المساعد العربي الذكي")
st.write("اختر شخصية المساعد واكتب سؤالك بالعربية:")

# اختيار الشخصية
persona = st.radio("اختر شخصية المساعد:", ["رسمية سياسية", "تحليل علم نفس", "تعليم للأطفال"])

# إدخال السؤال
question = st.text_area("اكتب سؤالك هنا:", placeholder="مثلاً: ما معنى الذكاء الاصطناعي؟")

# زر الإرسال
if st.button("إرسال"):
    prompt = f"بأسلوب {persona}، أجب على السؤال التالي:\n{question}"
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
