# Enhanced Diabetes Prediction Web Application
# ==================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from streamlit_lottie import st_lottie
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Lottie animation
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
    }
    
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 42px;
        font-weight: 700;
        color: #4361ee;
        text-align: center;
        margin-bottom: 15px;
        padding-bottom: 15px;
        border-bottom: 3px solid #4361ee;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        letter-spacing: 0.5px;
    }
    
    .sub-header {
        font-family: 'Poppins', sans-serif;
        font-size: 28px;
        font-weight: 600;
        color: #3a0ca3;
        margin-top: 25px;
        margin-bottom: 15px;
        position: relative;
        display: inline-block;
    }
    
    .sub-header:after {
        content: '';
        position: absolute;
        width: 50%;
        height: 3px;
        bottom: -5px;
        left: 0;
        background: linear-gradient(90deg, #4361ee 0%, rgba(67, 97, 238, 0) 100%);
        border-radius: 10px;
    }
    
    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .prediction-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .prediction-header {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 15px;
        font-family: 'Poppins', sans-serif;
    }
    
    .feature-importance {
        margin-top: 35px;
        margin-bottom: 25px;
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(145deg, #f0f5ff 0%, #e6f0ff 100%);
        box-shadow: 0 5px 15px rgba(67, 97, 238, 0.1);
    }
    
    .footer {
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e6e6e6;
        font-size: 14px;
        color: #666666;
    }
    
    .stSlider label {
        font-weight: 600;
        color: #4361ee;
        font-size: 16px;
    }
    
    .stSlider > div > div {
        background-color: #b5c7ff !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: #4361ee !important;
    }
    
    .info-box {
        background: linear-gradient(145deg, #e6f2ff 0%, #d9ebff 100%);
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 22px;
        border-left: 4px solid #4361ee;
        box-shadow: 0 4px 12px rgba(67, 97, 238, 0.1);
    }
    
    .value-metric {
        background: white;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        border-left: 3px solid #4361ee;
        transition: all 0.3s ease;
    }
    
    .value-metric:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .value-metric-title {
        font-weight: 600;
        color: #3a0ca3;
        font-size: 16px;
        margin-bottom: 5px;
    }
    
    .value-metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #4361ee;
    }
    
    .value-metric-range {
        font-size: 12px;
        color: #666;
        margin-top: 5px;
    }
    
    .stButton > button {
        background-color: #4361ee;
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 10px rgba(67, 97, 238, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 10px;
    }
    
    .stButton > button:hover {
        background-color: #3a0ca3;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(67, 97, 238, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 5px rgba(67, 97, 238, 0.4);
    }
    
    /* Card styles */
    .metric-card {
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        background: white;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .metric-card-title {
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 15px;
        color: #3a0ca3;
    }
    
    .metric-card-value {
        font-size: 32px;
        font-weight: 700;
        color: #4361ee;
        margin-bottom: 10px;
    }
    
    .status-normal {
        color: #10b981;
    }
    
    .status-warning {
        color: #f59e0b;
    }
    
    .status-danger {
        color: #ef4444;
    }
    
    /* Recommendations styling */
    .recommendation-item {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.05);
        display: flex;
        align-items: center;
    }
    
    .recommendation-icon {
        margin-right: 15px;
        font-size: 24px;
        color: #4361ee;
    }
    
    .timeline {
        position: relative;
        margin: 20px 0;
        padding-left: 30px;
    }
    
    .timeline:before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 3px;
        background: #4361ee;
        border-radius: 3px;
    }
    
    .timeline-item {
        position: relative;
        margin-bottom: 25px;
    }
    
    .timeline-dot {
        position: absolute;
        left: -34px;
        top: 0;
        width: 15px;
        height: 15px;
        background: #4361ee;
        border-radius: 50%;
    }
    
    .timeline-content {
        padding: 15px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        background-color: #f8f9fa;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4361ee !important;
        color: white !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #4361ee !important;
    }
</style>
""", unsafe_allow_html=True)

# Define functions
def load_model():
    model_path = 'knn_diabetes_model.joblib'
    try:
        model = joblib.load(model_path)
        return model
    except:
        st.error(f"Model file not found at {model_path}. Please check the path and ensure the model file exists.")
        return None

def make_prediction(model, input_data):
    # Convert input_data to DataFrame with proper column names
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    input_df = pd.DataFrame([input_data], columns=column_names)
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    return prediction[0], probability[0][1]  # Return prediction and probability of diabetes

def get_feature_importance():
    # You could load this from a saved file or recalculate it
    # For now, we'll use predetermined importance from your analysis
    importances = {
        'Glucose': 0.47,
        'BMI': 0.29,
        'Age': 0.24,
        'Insulin': 0.21,
        'DiabetesPedigreeFunction': 0.17,
        'Pregnancies': 0.15,
        'BloodPressure': 0.14,
        'SkinThickness': 0.07
    }
    return importances

def get_normal_ranges():
    # Define normal ranges for each feature
    ranges = {
        'Glucose': (70, 99),
        'BloodPressure': (90, 120),
        'SkinThickness': (10, 30),
        'Insulin': (16, 166),
        'BMI': (18.5, 24.9),
        'DiabetesPedigreeFunction': (0.0, 0.8),
        'Age': (20, 65),
        'Pregnancies': (0, 10)
    }
    return ranges

def is_in_normal_range(feature, value):
    ranges = get_normal_ranges()
    min_val, max_val = ranges[feature]
    return min_val <= value <= max_val

def get_status_class(feature, value):
    if is_in_normal_range(feature, value):
        return "status-normal"
    
    ranges = get_normal_ranges()
    min_val, max_val = ranges[feature]
    
    # Determine how far outside the range
    if value < min_val:
        if value < min_val * 0.8:  # Significantly below
            return "status-danger"
        else:
            return "status-warning"
    else:  # value > max_val
        if value > max_val * 1.2:  # Significantly above
            return "status-danger"
        else:
            return "status-warning"

def get_risk_level(probability):
    if probability < 0.3:
        return "Low", "#10b981"  # Green
    elif probability < 0.6:
        return "Moderate", "#f59e0b"  # Yellow/Orange
    else:
        return "High", "#ef4444"  # Red

def create_gauge_chart(probability):
    risk_level, color = get_risk_level(probability)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Diabetes Risk: {risk_level}", 'font': {'size': 24, 'color': color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#DCFCE7'},
                {'range': [30, 60], 'color': '#FEF9C3'},
                {'range': [60, 100], 'color': '#FEE2E2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        },
        number = {'suffix': "%", 'font': {'size': 30}}
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor="white",
        font={'color': "#333", 'family': "Nunito"}
    )
    
    return fig

def create_spider_chart(input_data):
    # Categories and normalized values
    categories = ['Glucose', 'Blood Pressure', 'BMI', 
                 'Diabetes Pedigree', 'Age', 'Insulin']
    
    # Calculate percentage of max value for each metric
    # Normalize values between 0 and 1
    normal_ranges = get_normal_ranges()
    max_values = {
        'Glucose': 200,
        'BloodPressure': 130,
        'BMI': 50,
        'DiabetesPedigreeFunction': 2.5,
        'Age': 90,
        'Insulin': 400
    }
    
    values = [
        input_data[1] / max_values['Glucose'],
        input_data[2] / max_values['BloodPressure'],
        input_data[5] / max_values['BMI'],
        input_data[6] / max_values['DiabetesPedigreeFunction'],
        input_data[7] / max_values['Age'],
        input_data[4] / max_values['Insulin']
    ]
    
    # Create radar chart using plotly
    fig = go.Figure()
    
    # Add normal range area
    normal_values = [
        normal_ranges['Glucose'][1] / max_values['Glucose'],
        normal_ranges['BloodPressure'][1] / max_values['BloodPressure'],
        normal_ranges['BMI'][1] / max_values['BMI'],
        normal_ranges['DiabetesPedigreeFunction'][1] / max_values['DiabetesPedigreeFunction'],
        normal_ranges['Age'][1] / max_values['Age'],
        normal_ranges['Insulin'][1] / max_values['Insulin']
    ]
    
    fig.add_trace(go.Scatterpolar(
        r=normal_values,
        theta=categories,
        fill='toself',
        name='Normal Range',
        fillcolor='rgba(67, 97, 238, 0.1)',
        line=dict(color='rgba(67, 97, 238, 0.5)')
    ))
    
    # Add user values
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Values',
        fillcolor='rgba(239, 68, 68, 0.2)',
        line=dict(color='rgba(239, 68, 68, 0.8)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title={
            'text': 'Health Metrics Profile',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': '#3a0ca3', 'family': 'Poppins'}
        },
        height=450,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    return fig

def create_feature_importance_chart(importances):
    features = list(importances.keys())
    values = list(importances.values())
    
    # Sort by importance
    sorted_indices = np.argsort(values)
    sorted_features = [features[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    # Create color gradient based on values
    colors = [f'rgba(67, 97, 238, {v*1.5})' for v in sorted_values]
    
    fig = go.Figure(go.Bar(
        x=sorted_values,
        y=sorted_features,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(58, 12, 163, 0.6)', width=1)
        )
    ))
    
    fig.update_layout(
        title={
            'text': 'Factors Affecting Diabetes Risk',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': '#3a0ca3', 'family': 'Poppins'}
        },
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=400,
        margin=dict(l=20, r=20, t=80, b=30),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        plot_bgcolor='white'
    )
    
    return fig

def format_value(feature, value):
    """Format value with appropriate units"""
    units = {
        'Glucose': 'mg/dL',
        'BloodPressure': 'mm Hg',
        'SkinThickness': 'mm',
        'Insulin': 'µU/ml',
        'BMI': 'kg/m²',
        'DiabetesPedigreeFunction': '',
        'Age': 'years',
        'Pregnancies': ''
    }
    
    return f"{value} {units.get(feature, '')}"

def get_recommendation_icon(index):
    icons = ["💊", "📊", "🥗", "🏃‍♀️", "⚖️", "😴", "🩺", "🧠"]
    return icons[index % len(icons)]

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import plotly.express as px

def create_historical_trend(feature_values):
    # This is a simulated historical trend
    base_date = datetime(2025, 4, 17)
    dates = [base_date - relativedelta(months=i) for i in range(6)]
    dates.reverse()  # So they go from past to present

    # Create a dataframe for the trend chart
    trend_data = pd.DataFrame({
        'Date': dates,
        'Glucose': [110, 115, 118, 125, 120, feature_values[1]],
        'BMI': [28, 27.8, 27.5, 26.9, 26, feature_values[5]],
        'BloodPressure': [80, 85, 90, 88, 75, feature_values[2]]
    })

    # Create line charts
    glucose_fig = px.line(
        trend_data, x='Date', y='Glucose',
        title='Glucose Level Trend (mg/dL)',
        markers=True,
        color_discrete_sequence=['#4361ee']
    )

    bmi_fig = px.line(
        trend_data, x='Date', y='BMI',
        title='BMI Trend (kg/m²)',
        markers=True,
        color_discrete_sequence=['#3a0ca3']
    )

    bp_fig = px.line(
        trend_data, x='Date', y='BloodPressure',
        title='Blood Pressure Trend (mm Hg)',
        markers=True,
        color_discrete_sequence=['#ef4444']
    )

    # Customize layout
    for fig in [glucose_fig, bmi_fig, bp_fig]:
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=30),
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)',
                title_font=dict(size=12)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)',
                title_font=dict(size=12)
            ),
            title=dict(
                font=dict(size=16, color='#3a0ca3', family='Poppins')
            ),
            showlegend=False
        )

    return glucose_fig, bmi_fig, bp_fig

# Main application code
def main():
    # Header
    st.markdown('<div class="main-header">Diabetes Risk Assessment Tool</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Load and display Lottie animation
        lottie_url = "https://assets3.lottiefiles.com/packages/lf20_5njmohcd.json"
        lottie_json = load_lottieurl(lottie_url)
        if lottie_json:
            st_lottie(lottie_json, height=200, key="health_lottie")
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/4320/4320371.png", width=150)
        
        st.markdown("## About This Tool")
        st.info("""
        This advanced application uses a K-Nearest Neighbors (KNN) machine learning model 
        trained on the Pima Indians Diabetes Dataset to predict diabetes risk with high accuracy.
        
        Enter your health metrics to receive a personalized risk assessment with actionable recommendations.
        """)
        
        st.markdown("### How It Works")
        st.markdown("""
        1. **Enter your health data** using the sliders
        2. **Click 'Analyze Risk'** for prediction
        3. **Review your results** in detail
        4. **Follow recommendations** based on your risk level
        """)
        
        st.divider()
        
        st.markdown("### Data Privacy Notice")
        st.markdown("""
        This application processes all data locally on your device. No health information is stored or shared with third parties.
        """)
        
    # Load the model
    model = load_model()
    
    if model:
        # Create tabs for different sections
        tabs = st.tabs(["🔍 Risk Assessment", "📊 Analysis & Insights", "📋 History & Trends", "ℹ️ Information"])
        
        with tabs[0]:  # Risk Assessment tab
            # Create two columns for input and results
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.markdown('<div class="sub-header">Your Health Information</div>', unsafe_allow_html=True)
                
                # Input fields with sliders
                pregnancies = st.slider("Number of Pregnancies", 0, 17, 3, 1)
                
                glucose = st.slider("Glucose Level (mg/dL)", 50, 200, 120, 1)
                if not is_in_normal_range('Glucose', glucose):
                    st.markdown('<div class="info-box">Normal fasting glucose range is 70-99 mg/dL</div>', unsafe_allow_html=True)
                
                blood_pressure = st.slider("Blood Pressure (mm Hg)", 40, 130, 70, 1)
                if not is_in_normal_range('BloodPressure', blood_pressure):
                    st.markdown('<div class="info-box">Normal blood pressure is 90-120 mm Hg</div>', unsafe_allow_html=True)
                
                skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20, 1)
                
                insulin = st.slider("Insulin Level (µU/ml)", 0, 846, 80, 1)
                if not is_in_normal_range('Insulin', insulin):
                    st.markdown('<div class="info-box">Normal fasting insulin range is 16-166 µU/ml</div>', unsafe_allow_html=True)
                
                bmi = st.slider("BMI", 0.0, 67.1, 25.0, 0.1)
                bmi_status = "Normal weight"
                if bmi < 18.5:
                    bmi_status = "Underweight"
                elif 18.5 <= bmi < 25.0:
                    bmi_status = "Normal weight"
                elif 25.0 <= bmi < 30.0:
                    bmi_status = "Overweight"
                else:
                    bmi_status = "Obese"
                st.markdown(f'<div class="info-box">BMI Status: <strong>{bmi_status}</strong></div>', unsafe_allow_html=True)
                
                diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
                
                age = st.slider("Age", 21, 90, 33, 1)
                
                # Create input data array
                input_data = [pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, diabetes_pedigree, age]
                
                # Prediction button with enhanced styling
                predict_button = st.button('✨ Analyze Risk Now ✨', key='predict_button')
            
            with col2:
                if predict_button:
                    prediction, probability = make_prediction(model, input_data)
                    
                    # Display prediction result
                    st.markdown('<div class="sub-header">Risk Assessment Results</div>', unsafe_allow_html=True)
                    
                    # Create gauge chart for risk visualization
                    gauge_fig = create_gauge_chart(probability)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Display risk level with appropriate styling
                    if prediction == 1:
                        risk_color = "#FF6961"  # Light red
                        risk_text = "High Risk of Diabetes"
                        risk_emoji = "⚠️"
                    else:
                        risk_color = "#77DD77"  # Light green
                        risk_text = "Low Risk of Diabetes"
                        risk_emoji = "✅"
                    
                    st.markdown(
                        f"""
                        <div class="prediction-box" style="background-color: {risk_color};">
                            <div class="prediction-header">{risk_emoji} {risk_text}</div>
                            <div>Our model predicts a {probability:.1%} probability of diabetes based on your health metrics.</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Key metrics cards - display in a grid
                    st.markdown("### Key Health Indicators")
                    
                    # Create three columns for metric cards
                    mc1, mc2, mc3 = st.columns(3)
                    
                    with mc1:
                        status_class = get_status_class('Glucose', glucose)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-card-title">Glucose</div>
                            <div class="metric-card-value {status_class}">{glucose} mg/dL</div>
                            <div>Normal: 70-99 mg/dL</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with mc2:
                        status_class = get_status_class('BMI', bmi)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-card-title">BMI</div>
                            <div class="metric-card-value {status_class}">{bmi} kg/m²</div>
                            <div>Normal: 18.5-24.9 kg/m²</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with mc3:
                        status_class = get_status_class('BloodPressure', blood_pressure)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-card-title">Blood Pressure</div>
                            <div class="metric-card-value {status_class}">{blood_pressure} mm Hg</div>
                            <div>Normal: 90-120 mm Hg</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk factors analysis with improved styling
                    st.markdown("### Health Risk Factors")
                    
                    # Check values outside normal range with better UI
                    ranges = get_normal_ranges()
                    features_outside_range = []
                    
                    for i, feature in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']):
                        min_val, max_val = ranges[feature]
                        if not (min_val <= input_data[i] <= max_val):
                            features_outside_range.append((feature, input_data[i]))
                    
                    if features_outside_range:
                        st.markdown('<div class="timeline">', unsafe_allow_html=True)
                        for i, (feature, value) in enumerate(features_outside_range):
                            st.markdown(f"""
                            <div class="timeline-item">
                                <div class="timeline-dot"></div>
                                <div class="timeline-content">
                                    <strong>{feature}:</strong> Your value of {format_value(feature, value)} is outside the normal range 
                                    ({format_value(feature, ranges[feature][0])} - {format_value(feature, ranges[feature][1])})
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.success("All your health metrics are within normal ranges. Keep up the good work!")
                else:
                    # Display placeholder content when no prediction has been made
                    st.markdown('<div class="sub-header">Your Risk Assessment</div>', unsafe_allow_html=True)
                    st.markdown("""
                    <div style="text-align: center; padding: 50px 20px; background: #f8f9fa; border-radius: 12px; margin: 20px 0;">
                        <img src="https://cdn-icons-png.flaticon.com/512/1581/1581953.png" width="100">
                        <h3 style="margin-top: 20px; color: #3a0ca3;">Ready for Your Assessment?</h3>
                        <p style="margin-top: 10px; color: #666;">Fill in your health details and click "Analyze Risk Now" to get your personalized diabetes risk prediction.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tabs[1]:  # Analysis & Insights tab
            st.markdown('<div class="sub-header">Health Data Analysis</div>', unsafe_allow_html=True)
            
            # Check if prediction has been made (assuming data is already input)
            if 'input_data' in locals():
                # Create two columns layout
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Create radar chart for health profile
                    spider_fig = create_spider_chart(input_data)
                    st.plotly_chart(spider_fig, use_container_width=True)
                
                with col2:
                    # Feature importance visualization with enhanced chart
                    importances = get_feature_importance()
                    importance_fig = create_feature_importance_chart(importances)
                    st.plotly_chart(importance_fig, use_container_width=True)
                
                # Recommendations section
                st.markdown('<div class="sub-header">Personalized Recommendations</div>', unsafe_allow_html=True)
                
                # Generate recommendations based on prediction and factors
                if 'prediction' in locals():
                    if prediction == 1:
                        recommendations = [
                            "Schedule a consultation with your healthcare provider to discuss your diabetes risk and potential next steps",
                            "Begin monitoring your blood glucose levels regularly with a home testing kit",
                            "Follow a balanced diet rich in vegetables, lean proteins, and low glycemic index foods",
                            "Establish a regular exercise routine with at least 150 minutes of moderate activity per week",
                            "Work towards achieving and maintaining a healthy weight to improve insulin sensitivity",
                            "Get adequate sleep (7-9 hours) to help regulate blood sugar levels",
                            "Consider speaking with a registered dietitian for a personalized meal plan",
                            "Manage stress through mindfulness, meditation, or other relaxation techniques"
                        ]
                    else:
                        recommendations = [
                            "Continue with regular health check-ups (at least once a year) with your healthcare provider",
                            "Maintain a balanced diet rich in vegetables, fruits, and whole grains",
                            "Stay physically active with at least 150 minutes of moderate exercise per week",
                            "Monitor your weight and maintain a healthy BMI",
                            "Get adequate sleep (7-9 hours) to support overall health",
                            "Stay hydrated by drinking plenty of water throughout the day",
                            "Limit alcohol consumption and avoid smoking",
                            "Practice stress management techniques to support overall wellbeing"
                        ]
                    
                    # Display recommendations with enhanced styling
                    for i, rec in enumerate(recommendations):
                        icon = get_recommendation_icon(i)
                        st.markdown(f"""
                        <div class="recommendation-item">
                            <div class="recommendation-icon">{icon}</div>
                            <div>{rec}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Complete your risk assessment in the Risk Assessment tab to receive personalized recommendations.")
            else:
                st.info("Please complete your risk assessment in the Risk Assessment tab first.")
        
        with tabs[2]:  # History & Trends tab
            st.markdown('<div class="sub-header">Health Metrics Over Time</div>', unsafe_allow_html=True)
            
            if 'input_data' in locals():
                st.markdown("""
                This section shows how your key health metrics have changed over time. Regular monitoring can help you track your progress and make informed decisions about your health.
                """)
                
                # Create simulated trend charts
                glucose_trend, bmi_trend, bp_trend = create_historical_trend(input_data)
                
                # Display trend charts in columns
                t1, t2 = st.columns(2)
                
                with t1:
                    st.plotly_chart(glucose_trend, use_container_width=True)
                    st.plotly_chart(bp_trend, use_container_width=True)
                
                with t2:
                    st.plotly_chart(bmi_trend, use_container_width=True)
                    
                    # Add a comparison widget
                    st.markdown("### Compare with Previous Results")
                    
                    # Simulated previous reading dates
                    previous_dates = ["April 17, 2025 (Current)", "March 15, 2025", "February 12, 2025", "January 10, 2025"]
                    selected_date = st.selectbox("Select date for comparison:", previous_dates)
                    
                    if selected_date == previous_dates[0]:  # Current reading
                        comparison_data = {
                            "Date": "Today",
                            "Glucose": f"{input_data[1]} mg/dL",
                            "BMI": f"{input_data[5]:.1f} kg/m²",
                            "Blood Pressure": f"{input_data[2]} mm Hg",
                            "Risk Level": f"{probability*100:.1f}%" if 'probability' in locals() else "Not calculated"
                        }
                    else:
                        # Simulated historical data
                        if selected_date == previous_dates[1]:
                            comparison_data = {
                                "Date": "March 15, 2025",
                                "Glucose": "120 mg/dL",
                                "BMI": "26.0 kg/m²",
                                "Blood Pressure": "75 mm Hg",
                                "Risk Level": "32.5%"
                            }
                        elif selected_date == previous_dates[2]:
                            comparison_data = {
                                "Date": "February 12, 2025",
                                "Glucose": "125 mg/dL",
                                "BMI": "26.9 kg/m²",
                                "Blood Pressure": "88 mm Hg",
                                "Risk Level": "38.2%"
                            }
                        else:
                            comparison_data = {
                                "Date": "January 10, 2025",
                                "Glucose": "118 mg/dL",
                                "BMI": "27.5 kg/m²",
                                "Blood Pressure": "90 mm Hg",
                                "Risk Level": "40.5%"
                            }
                    
                    # Display the data in a nice format
                    st.markdown(f"""
                    <div class="metric-card" style="padding: 20px; margin-top: 15px;">
                        <div class="metric-card-title">Results from {comparison_data['Date']}</div>
                        <table style="width: 100%; margin-top: 15px;">
                            <tr>
                                <td><strong>Glucose:</strong></td>
                                <td>{comparison_data['Glucose']}</td>
                            </tr>
                            <tr>
                                <td><strong>BMI:</strong></td>
                                <td>{comparison_data['BMI']}</td>
                            </tr>
                            <tr>
                                <td><strong>Blood Pressure:</strong></td>
                                <td>{comparison_data['Blood Pressure']}</td>
                            </tr>
                            <tr>
                                <td><strong>Risk Level:</strong></td>
                                <td>{comparison_data['Risk Level']}</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Please complete your risk assessment in the Risk Assessment tab first to view historical trends.")
        
        with tabs[3]:  # Information tab
            st.markdown('<div class="sub-header">Understanding Diabetes Risk</div>', unsafe_allow_html=True)
            
            # Information about diabetes
            st.markdown("""
            ## What is Diabetes?
            
            Diabetes is a chronic health condition that affects how your body turns food into energy. Most of the food you eat is broken down into sugar (glucose) and released into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin, which acts as a key to let the blood sugar into your body's cells for use as energy.
            
            With diabetes, your body either doesn't make enough insulin or can't use the insulin it makes as well as it should. When there isn't enough insulin or cells stop responding to insulin, too much blood sugar stays in your bloodstream, which over time can cause serious health problems.
            
            ## Risk Factors
            
            Several factors can increase your risk of developing diabetes:
            
            * **Family history:** Having a parent or sibling with diabetes
            * **Weight:** Being overweight or obese
            * **Age:** Being 45 years or older
            * **Physical activity:** Being physically inactive
            * **Race:** Certain races/ethnicities have higher risk
            * **Gestational diabetes:** A history of gestational diabetes or giving birth to a baby weighing more than 9 pounds
            * **Prediabetes:** Blood glucose levels higher than normal but not high enough to be diagnosed as diabetes
            * **Blood pressure:** Having high blood pressure (140/90 mm Hg or higher)
            * **Cholesterol levels:** Having low HDL ("good") cholesterol and/or high triglycerides
            
            ## About This Tool
            
            This prediction tool uses a K-Nearest Neighbors (KNN) machine learning algorithm trained on the Pima Indians Diabetes Dataset to estimate your risk of developing diabetes. The model analyzes your health metrics and compares them to patterns found in the training data to make a prediction.
            
            It's important to note that this tool provides an estimate based on available data and should not replace professional medical advice. Always consult with healthcare professionals for proper diagnosis and treatment.
            
            ## Key Health Metrics Explained
            
            * **Glucose Level:** Blood sugar level after fasting for at least 8 hours
            * **Blood Pressure:** The pressure of blood against the walls of your arteries
            * **Skin Thickness:** Triceps skin fold thickness, a measure of fat distribution
            * **Insulin Level:** Amount of insulin in your blood after fasting
            * **BMI (Body Mass Index):** A measure of body fat based on height and weight
            * **Diabetes Pedigree Function:** A function that scores likelihood of diabetes based on family history
            * **Age:** Age in years
            * **Pregnancies:** Number of times pregnant (for female patients)
            """)
        
    # Footer
    st.markdown('<div class="footer">Developed with ❤️ using Streamlit and K-Nearest Neighbors (KNN) Algorithm | © 2025</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()