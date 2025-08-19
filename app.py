# streamlit_app.py

import streamlit as st
import joblib
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Personalized Learning Assistant",
    page_icon="🧠",
    layout="wide"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('student_model_v2.pkl')
        print("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error("❌ Error: 'student_model_v2.pkl' not found. Please ensure the model file is in the same directory.")
        return None

MODEL = load_model()

# --- PREDICTION FUNCTION (Unchanged) ---
def personalize_learning_path(quiz_1, quiz_2, engagement, concepts, avg_time):
    if MODEL is None:
        return "Model not loaded.", {}

    columns = ['quiz_1_score', 'quiz_2_score', 'platform_engagement_days', 'concepts_mastered', 'avg_time_per_question']
    input_data = pd.DataFrame([[quiz_1, quiz_2, engagement, concepts, avg_time]], columns=columns)
    
    prediction_proba = MODEL.predict_proba(input_data)[0]
    will_struggle_prob = prediction_proba[0]
    
    recommendation = ""
    score_trend = quiz_2 - quiz_1

    if will_struggle_prob > 0.5:
        recommendation = "⚠️ **Prediction: High Risk of Struggling.**\n\n"
        if score_trend < -10:
            recommendation += "**Analysis:** Your quiz scores are declining significantly.\n**Action:** Let's pause new topics and revisit the material from the first quiz to fix the foundational gap."
        elif concepts < 50:
            recommendation += "**Analysis:** You seem to be struggling with the core concepts.\n**Action:** Let's switch to a conceptual review with 'why' and 'how' questions."
        elif avg_time < 30:
            recommendation += "**Analysis:** You are answering questions extremely quickly, suggesting guessing.\n**Action:** For the next module, let's focus on accuracy over speed."
        else:
            recommendation += "**Analysis:** There seems to be a general difficulty with the recent material.\n**Action:** I recommend a 15-minute review session covering the last three topics."
    else:
        recommendation = "✅ **Prediction: Ready to Advance!**\n\n"
        if score_trend > 10:
            recommendation += "**Analysis:** Fantastic improvement! Your scores are trending upwards.\n**Action:** You've earned a 'fast-track' token. You can skip the next introductory video and jump to the advanced challenge."
        else:
            recommendation += "**Analysis:** You are maintaining a solid and consistent understanding.\n**Action:** Let's keep the momentum going. The next module is ready for you."

    confidence_label = {"Struggle": will_struggle_prob, "Succeed": prediction_proba[1]}
    return recommendation, confidence_label

# --- STREAMLIT UI ---
st.title("🧠 AI Personalized Learning Assistant")
st.markdown("This advanced AI tutor analyzes deep learning patterns to provide hyper-personalized recommendations.")

st.divider()

with st.sidebar:
    st.header("Student Performance Metrics")
    quiz_1 = st.slider("Quiz 1 Score (%)", 0, 100, 70)
    quiz_2 = st.slider("Quiz 2 Score (%)", 0, 100, 80, help="Your most recent score.")
    engagement = st.slider("Platform Engagement (Days)", 1, 30, 15)
    concepts = st.slider("Core Concepts Mastered (%)", 0, 100, 55)
    avg_time = st.slider("Avg. Time Per Question (Seconds)", 10, 180, 45)

if MODEL is not None:
    recommendation, confidence = personalize_learning_path(quiz_1, quiz_2, engagement, concepts, avg_time)
    
    st.subheader("Personalized Recommendation")
    st.markdown(recommendation)

    st.subheader("Prediction Confidence")
    prob_struggle = confidence.get("Struggle", 0.0)
    prob_succeed = confidence.get("Succeed", 0.0)
    
    st.markdown("**Probability of Struggling**")
    st.progress(prob_struggle, text=f"{prob_struggle:.0%}")
    
    st.markdown("**Probability of Succeeding**")
    st.progress(prob_succeed, text=f"{prob_succeed:.0%}")
