import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. إعدادات الصفحة الرئيسية
# ---------------------------------------------------------
st.set_page_config(
    page_title="لوحة تحليل البيانات",
    page_icon="📊",
    layout="wide"
)

st.title("📊 تطبيق تحليل وتصور البيانات")
st.write("مرحباً بك! قم برفع ملف CSV لرؤية التحليلات والرسوم البيانية بشكل تفاعلي.")

# ---------------------------------------------------------
# 2. القائمة الجانبية (Sidebar) ورفع الملفات
# ---------------------------------------------------------
st.sidebar.header("📁 إدارة البيانات")
uploaded_file = st.sidebar.file_uploader("اختر ملف CSV", type=["csv"])

# ---------------------------------------------------------
# 3. معالجة وعرض البيانات عند رفع الملف
# ---------------------------------------------------------
if uploaded_file is not None:
    # قراءة الملف
    df = pd.read_csv(uploaded_file)
    
    # تقسيم الصفحة إلى تبويبات (Tabs) لترتيب العرض
    tab1, tab2, tab3 = st.tabs(["📋 نظرة عامة على البيانات", "📈 الرسوم البيانية", "⚙️ تصفية البيانات"])

    # --- التبويب الأول: نظرة عامة ---
    with tab1:
        st.subheader("معاينة البيانات (Data Preview)")
        st.dataframe(df.head())

        # عرض معلومات سريعة في كروت (Metrics)
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("عدد الأسطر (Rows)", df.shape[0])
        col2.metric("عدد الأعمدة (Columns)", df.shape[1])
        col3.metric("عدد القيم المفقودة", df.isnull().sum().sum())

        st.subheader("الإحصاءات الوصفية (Summary Statistics)")
        st.write(df.describe())

    # --- التبويب الثاني: الرسوم البيانية ---
    with tab2:
        st.subheader("رسم بياني تفاعلي")

        # استخراج الأعمدة الرقمية والفئوية
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        all_cols = df.columns.tolist()

        if numeric_cols:
            col_x = st.selectbox("اختر المحور الأفقـي (X-axis):", all_cols)
            col_y = st.selectbox("اختر المحور الرأسـي (Y-axis):", numeric_cols)
            
            chart_type = st.radio("اختر نوع الرسم البياني:", ["Bar Chart", "Line Chart", "Scatter Plot"])

            # إنشاء الشكل
            fig, ax = plt.subplots(figsize=(8, 4))

            if chart_type == "Bar Chart":
                sns.barplot(data=df, x=col_x, y=col_y, ax=ax, palette="Blues_d")
            elif chart_type == "Line Chart":
                sns.lineplot(data=df, x=col_x, y=col_y, ax=ax, color="green")
            elif chart_type == "Scatter Plot":
                sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax, color="purple")

            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("⚠️ لا توجد أعمدة رقمية في هذا الملف لرسمها.")

    # --- التبويب الثالث: تصفية البيانات ---
    with tab3:
        st.subheader("فلترة البيانات حسب عمود معين")
        selected_col = st.selectbox("اختر العمود للفلترة:", df.columns)
        
        unique_values = df[selected_col].unique()
        selected_val = st.selectbox("اختر القيمة:", unique_values)

        filtered_df = df[df[selected_col] == selected_val]
        st.write(f"النتائج المفلترة ({len(filtered_df)} صفوف):")
        st.dataframe(filtered_df)

else:
    # رسالة تظهر في حال لم يتم رفع ملف بعد
    st.info("👈 من فضلك قم برفع ملف CSV من القائمة الجانبية للبدء.")