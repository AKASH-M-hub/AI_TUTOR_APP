# app.py (v3.0 - Enhanced Modern Theme)

import gradio as gr
import joblib
import pandas as pd

# --- 1. DEFINE THE ENHANCED CUSTOM THEME ---
# Vibrant modern theme with gradients, smooth buttons, and refined typography
modern_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.purple,      # vibrant accent
    secondary_hue=gr.themes.colors.fuchsia,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Roboto"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    # Backgrounds
    body_background_fill="#F0F3F7",
    body_background_fill_dark="#1E1E2F",
    panel_background_fill="#FFFFFF",
    panel_background_fill_dark="#2B2B3A",
    
    # Buttons
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
    button_secondary_background_fill="*secondary_400",
    button_secondary_background_fill_hover="*secondary_500",
    button_secondary_text_color="white",
    
    # Sliders
    slider_color="*primary_500",
    slider_color_dark="*primary_400",
    
    # Titles & Labels
    block_title_text_color="*primary_700",
    block_title_text_color_dark="*primary_300",
    block_label_text_color="*secondary_700",
    block_label_text_color_dark="*secondary_300"
)

# --- 2. LOAD THE MODEL ---
try:
    MODEL = joblib.load('student_model_v2.pkl')
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'student_model_v2.pkl' not found.")
    MODEL = None

# --- 3. PREDICTION & PERSONALIZATION FUNCTION ---
def personalize_learning_path(quiz_1, quiz_2, engagement, concepts, avg_time):
    if MODEL is None:
        return "Model not loaded. Check the console.", {}

    columns = ['quiz_1_score', 'quiz_2_score', 'platform_engagement_days', 'concepts_mastered', 'avg_time_per_question']
    input_data = pd.DataFrame([[quiz_1, quiz_2, engagement, concepts, avg_time]], columns=columns)
    
    prediction_proba = MODEL.predict_proba(input_data)[0]
    will_struggle_prob = prediction_proba[0]
    
    recommendation = ""
    score_trend = quiz_2 - quiz_1

    if will_struggle_prob > 0.5:
        recommendation = "⚠️ **High Risk of Struggling**\n\n"
        if score_trend < -10:
            recommendation += "**Trend:** Declining scores.\n**Action:** Review previous material before moving forward."
        elif concepts < 50:
            recommendation += "**Trend:** Core concepts not mastered.\n**Action:** Focus on conceptual understanding with targeted exercises."
        elif avg_time < 30:
            recommendation += "**Trend:** Quick guessing observed.\n**Action:** Slow down and prioritize accuracy."
        else:
            recommendation += "**Trend:** General difficulty detected.\n**Action:** Suggested 15-min review of last topics."
    else:
        recommendation = "✅ **Ready to Advance!**\n\n"
        if score_trend > 10:
            recommendation += "**Trend:** Improving scores!\n**Action:** Fast-track next module."
        else:
            recommendation += "**Trend:** Consistent understanding.\n**Action:** Proceed to next module."

    confidence_label = {"Struggle": will_struggle_prob, "Succeed": prediction_proba[1]}
    return recommendation, confidence_label

# --- 4. CREATE THE GRADIO INTERFACE ---
iface = gr.Interface(
    fn=personalize_learning_path,
    inputs=[
        gr.Slider(0, 100, value=70, label="Quiz 1 Score (%)"),
        gr.Slider(0, 100, value=80, label="Quiz 2 Score (%)"),
        gr.Slider(1, 30, value=15, label="Platform Engagement (Days)"),
        gr.Slider(0, 100, value=55, label="Core Concepts Mastered (%)"),
        gr.Slider(10, 180, value=45, label="Avg. Time Per Question (Seconds)")
    ],
    outputs=[
        gr.Textbox(label="Personalized Recommendation", lines=8),
        gr.Label(label="Prediction Confidence")
    ],
    title="🧠 AI Personalized Learning Assistant v3.0",
    description="Interactive AI tutor providing adaptive learning recommendations with ML-driven insights.",
    theme=modern_theme,
    live=True
)

# --- 5. LAUNCH ---
if __name__ == "__main__":
    iface.launch()
