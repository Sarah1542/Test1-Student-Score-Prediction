import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. تحميل الموديل والـ Scaler
with open("student_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 2. القائمة بالترتيب الصح (كما تم استخراجها من x_train.columns)
model_columns = [
    'Attendance', 'Hours_Studied', 'Previous_Scores', 'Tutoring_Sessions', 
    'Access_to_Resources_Low', 'Parental_Involvement_Low', 
    'Parental_Education_Level_Postgraduate', 'Parental_Education_Level_High School', 
    'Learning_Disabilities_Yes', 'Distance_from_Home_Near', 'Family_Income_Low', 
    'Peer_Influence_Positive', 'Motivation_Level_Low', 
    'Extracurricular_Activities_Yes', 'Internet_Access_Yes'
]

st.set_page_config(page_title="Student Performance", layout="centered")
st.title("🎓 Student Performance Predictor")

# واجهة المستخدم (Sliders)
study_hours = st.slider("Hours Studied (per day)", 0, 12, 7)
attendance = st.slider("Attendance (%)", 0, 100, 100)
sleep_hours = st.slider("Sleep Hours (per day)", 0, 12, 8) # السلايدر الجديد
previous_scores = st.slider("Previous Scores", 0, 100, 92)
assignments = st.slider("Assignments Completion (%)", 0, 100, 100)

if st.button("Predict Score"):
    # أ- إنشاء DataFrame فاضي بنفس ترتيب الـ Scaler (15 عمود فقط)
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # ب- تعبئة القيم الأساسية التي يفهمها الموديل
    input_df['Attendance'] = attendance
    input_df['Hours_Studied'] = study_hours
    input_df['Previous_Scores'] = previous_scores
    input_df['Tutoring_Sessions'] = assignments / 10 
    
    # ج- تفعيل "بيئة النجاح" للعوامل الباقية
    input_df['Internet_Access_Yes'] = 1
    input_df['Peer_Influence_Positive'] = 1
    input_df['Distance_from_Home_Near'] = 1
    
    # د- الـ Scaling (تحويل البيانات للغة الموديل)
    input_scaled = scaler.transform(input_df)
    
    # هـ- التوقع الأصلي من الموديل
    prediction = model.predict(input_scaled)
    base_score = prediction[0]

    # و- المعايرة الذكية (Calibration) - هنا يدخل تأثير ساعات النوم
    # لو الطالب مثالي وبيريح جسمه وبيركز، السكور بيطلع لمستويات عالية
    if study_hours >= 7 and attendance >= 95 and sleep_hours >= 7:
        # بونص "التركيز العالي" بسبب النوم الكافي
        final_score = base_score + (100 - base_score) * 0.8
    elif sleep_hours < 5:
        # خصم "الإجهاد" لو الطالب مش بينام كويس
        final_score = base_score - 5
    else:
        # تحسين بسيط للنتائج المتوسطة
        final_score = base_score + 5

    final_score = max(0, min(100, final_score))

    # 3. عرض النتيجة
    st.subheader("Predicted Score")
    st.success(f"Expected Grade: {final_score:.2f}%")
    st.progress(int(final_score))

    if final_score > 90:
        st.balloons()
        st.write("Excellent! Your balance between study and health is great 💪")
    elif final_score > 60:
        st.write("Good job! Keep it up 👍")
    else:
        st.write("Consider organizing your time to get more rest and study ")