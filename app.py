import streamlit as st
import joblib
import numpy as np
import os

model    = joblib.load('attrition_model.pkl')
scaler   = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

st.set_page_config(page_title="Employee Attrition Predictor",
                   page_icon="👔", layout="centered")

st.title("👔 Employee Attrition Predictor")
st.markdown("Predict whether an employee is likely to leave the company.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    Age                      = st.slider("Age", 18, 60, 30)
    MonthlyIncome            = st.number_input("Monthly Income", 1000, 20000, 5000)
    OverTime                 = st.selectbox("OverTime", [0, 1],
                               format_func=lambda x: "Yes" if x == 1 else "No")
    JobSatisfaction          = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    YearsAtCompany           = st.slider("Years at Company", 0, 40, 5)
    WorkLifeBalance          = st.slider("Work Life Balance (1-4)", 1, 4, 3)
    EnvironmentSatisfaction  = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)

with col2:
    DistanceFromHome         = st.slider("Distance from Home (km)", 1, 30, 10)
    NumCompaniesWorked       = st.slider("Companies Worked Before", 0, 9, 2)
    TotalWorkingYears        = st.slider("Total Working Years", 0, 40, 8)
    JobLevel                 = st.slider("Job Level (1-5)", 1, 5, 2)
    StockOptionLevel         = st.slider("Stock Option Level (0-3)", 0, 3, 1)
    YearsWithCurrManager     = st.slider("Years with Current Manager", 0, 17, 4)
    TrainingTimesLastYear    = st.slider("Trainings Last Year", 0, 6, 2)

st.divider()

if st.button("🔮 Predict Attrition Risk", use_container_width=True):
    # Build input with all features (fill defaults for unused ones)
    input_dict = {feat: 0 for feat in features}

    input_dict.update({
        'Age'                     : Age,
        'MonthlyIncome'           : MonthlyIncome,
        'OverTime'                : OverTime,
        'JobSatisfaction'         : JobSatisfaction,
        'YearsAtCompany'          : YearsAtCompany,
        'WorkLifeBalance'         : WorkLifeBalance,
        'EnvironmentSatisfaction' : EnvironmentSatisfaction,
        'DistanceFromHome'        : DistanceFromHome,
        'NumCompaniesWorked'      : NumCompaniesWorked,
        'TotalWorkingYears'       : TotalWorkingYears,
        'JobLevel'                : JobLevel,
        'StockOptionLevel'        : StockOptionLevel,
        'YearsWithCurrManager'    : YearsWithCurrManager,
        'TrainingTimesLastYear'   : TrainingTimesLastYear,
    })

    input_array  = np.array([[input_dict[f] for f in features]])
    input_scaled = scaler.transform(input_array)
    prob         = model.predict_proba(input_scaled)[0][1]
    label        = model.predict(input_scaled)[0]

    st.divider()
    if label == 1:
        st.error("⚠️ This employee is **LIKELY TO LEAVE**")
    else:
        st.success("✅ This employee is **LIKELY TO STAY**")

    st.metric("Attrition Probability", f"{prob:.1%}")
    st.progress(float(prob))

    st.subheader("💡 HR Recommendation")
    if prob > 0.7:
        st.warning("🚨 High Risk — Immediate action needed! Consider salary revision, role change, or flexible work options.")
    elif prob > 0.4:
        st.warning("🟡 Medium Risk — Schedule a 1-on-1 meeting. Review workload and career growth opportunities.")
    else:
        st.info("🟢 Low Risk — Employee is stable. Keep up regular engagement and recognition.")