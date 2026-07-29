# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')




from datasets import load_dataset

# تحميل الداتا من HuggingFace
dataset = load_dataset("bobez999/arabic-qa-dataset-sigir2024")

# عرض أول صف للتأكد
print(dataset['train'][0])


import pandas as pd

# تحويل الداتا إلى DataFrame
df = pd.DataFrame(dataset['train'])

# إزالة الرموز الغريبة أو النصوص الفارغة
df = df.dropna()
df['instruction'] = df['instruction'].str.replace('[^ء-ي ]', '', regex=True)
df['input'] = df['input'].str.replace('[^ء-ي ]', '', regex=True)
df['output'] = df['output'].str.replace('[^ء-ي ]', '', regex=True)

# عرض أول 5 صفوف بعد التنظيف
df.head()


!pip install transformers datasets sentence-transformers langchain faiss-cpu accelerate evaluate pydantic gradio
!pip install langchain langchain-community faiss-cpu sentence-transformers
!pip install -U langchain langchain-community langchain-huggingface langchain-core langchain-text-splitters
!pip install -U langchain langchain-community langchain-huggingface
!pip install -U langchain langchain-community langchain-huggingface langchain-core
!pip install -U langchain langchain-community langchain-huggingface langchain-core
!pip install -U huggingface_hub
!pip install -U huggingface_hub transformers accelerate


import transformers, datasets, sentence_transformers, langchain, faiss, accelerate, evaluate, pydantic, gradio
print("✅ كل المكتبات جاهزة!")



import gradio as gr

# دالة اختيار الشخصية
def choose_persona(persona, question):
    if persona == "رسمية سياسية":
        # هنا تربطي بمصادر سياسية موثوقة
        answer = f"📘 [سياسة] إجابة رسمية عن: {question}\nالمصدر: تقارير حكومية + مقالات إخبارية."
    elif persona == "تحليل علم نفس":
        # هنا تربطي بمصادر علم نفس وعلاقات
        answer = f"🧠 [علم نفس] تحليل مبسط لسؤالك: {question}\nالمصدر: كتب علم نفس + مقالات APA."
    elif persona == "تعليم للأطفال":
        # هنا تربطي بمصادر تعليمية للأطفال
        answer = f"🎓 [تعليم أطفال] شرح مبسط لسؤالك: {question}\nالمصدر: قصص تعليمية + محتوى Nafham."
    else:
        answer = "من فضلك اختر شخصية صحيحة."
    return answer

# قائمة الشخصيات
personas = ["رسمية سياسية", "تحليل علم نفس", "تعليم للأطفال"]

# واجهة Gradio
iface = gr.Interface(
    fn=choose_persona,
    inputs=[gr.Radio(personas, label="اختر شخصية المساعد"), gr.Textbox(label="اكتب سؤالك بالعربية")],
    outputs="text",
    title="المساعد العربي الذكي",
    description="اختر شخصية المساعد (سياسية، علم نفس، تعليم أطفال) واكتب سؤالك"
)

iface.launch()


from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS


# إنشاء نموذج لتحويل النصوص إلى Embeddings
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# أمثلة لمصادر نصية لكل تخصص
political_docs = ["تقارير حكومية", "مقالات سياسية", "دستور عربي"]
psychology_docs = ["كتب علم نفس", "مقالات عن العلاقات", "نصائح سلوكية"]
children_docs = ["قصص تعليمية", "دروس مبسطة", "أنشطة للأطفال"]


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# إنشاء كائن embeddings من LangChain
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# تحويل النصوص إلى متجهات وتخزينها
political_db = FAISS.from_texts(political_docs, embedding_model)
psychology_db = FAISS.from_texts(psychology_docs, embedding_model)
children_db = FAISS.from_texts(children_docs, embedding_model)



def get_retriever(persona):
    if persona == "رسمية سياسية":
        return political_db.as_retriever()
    elif persona == "تحليل علم نفس":
        return psychology_db.as_retriever()
    elif persona == "تعليم للأطفال":
        return children_db.as_retriever()



from huggingface_hub import InferenceClient

client = InferenceClient(
    "tiiuae/falcon-7b-instruct",
    token="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"
)


from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# تحميل الموديل من HuggingFace


client = InferenceClient(
    "tiiuae/falcon-7b-instruct",
    token="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"
)

# إنشاء Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# بناء قاعدة بيانات لكل تخصص
political_db = FAISS.from_texts(["تقارير حكومية", "مقالات سياسية", "دستور عربي"], embedding_model)
psychology_db = FAISS.from_texts(["كتب علم نفس", "مقالات عن العلاقات", "نصائح سلوكية"], embedding_model)
children_db = FAISS.from_texts(["قصص تعليمية", "دروس مبسطة", "أنشطة للأطفال"], embedding_model)

# دالة بحث بديلة
def ask_question(persona, question):
    # تحديد قاعدة البيانات حسب الشخصية
    if persona == "رسمية سياسية":
        retriever = political_db.as_retriever()
    elif persona == "تحليل علم نفس":
        retriever = psychology_db.as_retriever()
    elif persona == "تعليم للأطفال":
        retriever = children_db.as_retriever()
    else: 
        return "❌ شخصية غير معروفة"
    
        # استرجاع أقرب نصوص من القاعدة
        docs = retriever.invoke(question)
        context = " ".join([d.page_content for d in docs])
    
        # إنشاء الـ prompt
        prompt = f"السؤال: {question}\nالمصادر: {context}\nالإجابة:"
    
        # توليد النص باستخدام InferenceClient
        response = client.text_generation(prompt, max_new_tokens=200)
    
        return response




    import gradio as gr

import asyncio

def interface_function(persona, question):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(asyncio.to_thread(ask_question, persona, question))


personas = ["رسمية سياسية", "تحليل علم نفس", "تعليم للأطفال"]

iface = gr.Interface(
    fn=interface_function,
    inputs=[
        gr.Radio(personas, label="اختر شخصية المساعد"),
        gr.Textbox(label="اكتب سؤالك بالعربية")
    ],
    outputs="text",
    title="المساعد العربي الذكي",
    description="اختر شخصية المساعد واسأل سؤالك ليجيب من مصادر موثوقة"
)

iface.launch()


!pip install streamlit


import streamlit as st

st.title("المساعد العربي الذكي 🤖")
persona = st.radio("اختر شخصية المساعد:", ["رسمية سياسية", "تحليل علم نفس", "تعليم للأطفال"])
question = st.text_input("اكتب سؤالك بالعربية:")

if st.button("إرسال"):
    answer = ask_question(persona, question)
    st.write("### الإجابة:")
    st.write(answer)

