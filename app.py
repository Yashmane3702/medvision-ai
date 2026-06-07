import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import time

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MedVision AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Hide Streamlit Branding */

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Background */

.stApp{
background:
linear-gradient(
135deg,
#020617,
#0f172a,
#111827
);
color:white;
font-family: 'Poppins', sans-serif;
}

/* Hero Title */

.hero-title{
font-size:72px;
font-weight:800;
line-height:1;
background: linear-gradient(to right,#ff4b4b,#ff7b7b);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

/* Subtitle */

.hero-sub{
font-size:20px;
color:#cbd5e1;
margin-top:15px;
}

/* Glass Effect */

.glass{
background: rgba(255,255,255,0.05);
backdrop-filter: blur(20px);
border:1px solid rgba(255,255,255,0.08);
padding:30px;
border-radius:24px;
box-shadow:0 8px 32px rgba(0,0,0,0.3);
}

/* Input Labels */

.stSelectbox label,
.stSlider label,
.stNumberInput label{
color:white !important;
font-size:15px !important;
}

/* Button */

.stButton>button{
width:100%;
height:60px;
border:none;
border-radius:16px;
background:linear-gradient(90deg,#ff4b4b,#ff6b81);
color:white;
font-size:20px;
font-weight:700;
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.02);
box-shadow:0 0 25px rgba(255,75,75,0.5);
}

/* Result */

.result-text{
font-size:32px;
font-weight:700;
text-align:center;
}

/* Sidebar */

section[data-testid="stSidebar"]{
background-color:#0b1120;
border-right:1px solid rgba(255,255,255,0.08);
}

/* Metric Cards */

.metric-card{
background: rgba(255,255,255,0.05);
padding:16px;
border-radius:18px;
border:1px solid rgba(255,255,255,0.08);
margin-bottom:15px;
}

/* HR */

hr{
border:none;
height:1px;
background:rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(r"E:\ml projects\begineer\heart disease\knn_heart.pkl")
scaler = joblib.load(r"E:\ml projects\begineer\heart disease\scaler.pkl")
expected_columns = joblib.load(r"E:\ml projects\begineer\heart disease\columns.pkl")

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <h2 style='color:white;'>🫀 MedVision AI</h2>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Accuracy Card

    st.markdown("""
    <div class="metric-card">
        <p style="
            color:#94a3b8;
            margin:0;
            font-size:14px;
        ">
        Model Accuracy
        </p>

        <h3 style="
            color:white;
            margin-top:5px;
        ">
        87%
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # Model Card

    st.markdown("""
    <div class="metric-card">
        <p style="
            color:#94a3b8;
            margin:0;
            font-size:14px;
        ">
        ML Algorithm
        </p>

        <h3 style="
            color:white;
            margin-top:5px;
        ">
        KNN Classifier
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # Status Card

    st.markdown("""
    <div class="metric-card">
        <p style="
            color:#94a3b8;
            margin:0;
            font-size:14px;
        ">
        AI Status
        </p>

        <h3 style="
            color:#22c55e;
            margin-top:5px;
        ">
        ● Active
        </h3>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="glass">

<div class="hero-title">
MedVision AI
</div>

<div class="hero-sub">
AI-powered cardiovascular disease prediction system with real-time health intelligence and premium analytics dashboard.
</div>

</div>
""", unsafe_allow_html=True)

st.write("")
st.write("")

# =========================================================
# INPUT SECTION
# =========================================================

st.markdown('<div class="glass">', unsafe_allow_html=True)

st.subheader("🩺 Patient Health Information")

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 18, 100, 40)

    sex = st.selectbox(
        "Gender",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        80, 200, 120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        100, 600, 200
    )

with col2:

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120",
        [0, 1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60, 220, 150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak",
        0.0, 6.0, 1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

predict = st.button("🚀 Analyze Heart Risk")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PREDICTION
# =========================================================

if predict:

    with st.spinner("Analyzing cardiovascular patterns..."):
        time.sleep(2)

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Fill missing columns

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    # Scale

    scaled_input = scaler.transform(input_df)

    # Prediction

    prediction = model.predict(scaled_input)[0]

    probability = model.predict_proba(scaled_input)[0][1]

    st.write("")
    st.write("")

    # =====================================================
    # RESULT SECTION
    # =====================================================

    result1, result2 = st.columns(2)

    # LEFT CARD

    with result1:

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        if prediction == 1:

            st.markdown("""
            <div class="result-text">
            ⚠️ High Risk Detected
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-text">
            ✅ Low Risk Detected
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        st.progress(int(probability * 100))

        st.markdown(
            f"""
            <h1 style='text-align:center; color:white;'>
            {probability*100:.1f}%
            </h1>
            """,
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT CARD

    with result2:

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Heart Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ff4b4b"},
                'steps': [
                    {'range': [0, 30], 'color': "#22c55e"},
                    {'range': [30, 70], 'color': "#facc15"},
                    {'range': [70, 100], 'color': "#ef4444"}
                ]
            }
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "white"}
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # HEALTH INSIGHTS
    # =====================================================

    st.write("")
    st.write("")

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("📊 AI Health Insights")

    if prediction == 1:

        st.warning("""
        Recommendations:

        • Improve cardiovascular fitness  
        • Reduce cholesterol intake  
        • Exercise regularly  
        • Reduce stress levels  
        • Consult a cardiologist  
        """)

    else:

        st.success("""
        Your cardiovascular indicators appear healthy.

        Maintain:

        • Balanced diet  
        • Daily exercise  
        • Good sleep cycle  
        • Regular medical checkups  
        """)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.write("")
st.write("")

st.markdown("""
<center style='color:#94a3b8'>
Made with ❤️ by <b>Yash Mane</b><br>
AI Healthcare Intelligence Platform
</center>
""", unsafe_allow_html=True)