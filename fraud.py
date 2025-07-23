import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("college_student_placement_dataset.csv")
    df = df.drop(columns=["College_ID"])
    df.dropna(inplace=True)
    df["Internship_Experience"] = df["Internship_Experience"].map({"Yes": 1, "No": 0})
    df["Placement"] = df["Placement"].map({"Yes": 1, "No": 0})
    return df

df = load_data()

# Features and target
X = df.drop("Placement", axis=1)
y = df["Placement"]

# Standardize input
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalance
sample_weights = class_weight.compute_sample_weight(class_weight="balanced", y=y)

# Train/test split
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X_scaled, y, sample_weights, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🎓 Student Placement Predictor")

st.markdown("### Enter student details to predict placement outcome.")

# User input form
with st.form("Placement Form"):
    iq = st.slider("IQ", 50, 160, 100)
    prev_sem = st.slider("Previous Semester Result", 4.0, 10.0, 7.0)
    cgpa = st.slider("CGPA", 4.0, 10.0, 7.0)
    acad_perf = st.slider("Academic Performance", 1, 10, 5)
    internship = st.selectbox("Internship Experience", options=["Yes", "No"])
    extra_curricular = st.slider("Extra-Curricular Score", 0, 10, 5)
    comm_skills = st.slider("Communication Skills", 0, 10, 5)
    projects = st.slider("Projects Completed", 0, 6, 2)
    
    submitted = st.form_submit_button("Predict")

# Predict
if submitted:
    features = np.array([
        iq, prev_sem, cgpa, acad_perf,
        1 if internship == "Yes" else 0,
        extra_curricular, comm_skills, projects
    ]).reshape(1, -1)

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][prediction]

    if prediction == 1:
        st.markdown(
        f"<div style='background-color: #d4edda; padding: 10px; border-radius: 5px; color: #155724;'>"
        f"<b>✅ will be Placed</b>"
        "</div>",
        unsafe_allow_html=True
    )
    else:
        st.markdown(
        f"<div style='background-color: #f8d7da; padding: 10px; border-radius: 5px; color: #721c24;'>"
        f"<b>❌ less chances To Place</b>"
        "</div>",
        unsafe_allow_html=True
    )

# Optional: Display metrics
with st.expander("Show Model Performance on Test Set"):
    y_pred = model.predict(X_test)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))


st.markdown("""
    <style>
    .stApp {
        background-color:#e6fff9; /* Light grey-blue */
    }
    </style>
""", unsafe_allow_html=True)
import streamlit as st

st.markdown(
    """
    <style>
    body {
        background-color: #e6f2ff;
        color: #000000;
    }
    .stApp {
        background-color: #e6f2ff;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
