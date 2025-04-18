# Diabetes Prediction Web Application
# ==================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #3366ff;
        text-align: center;
        margin-bottom: 10px;
        padding-bottom: 15px;
        border-bottom: 2px solid #3366ff;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #0033cc;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .prediction-header {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .feature-importance {
        margin-top: 30px;
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f5ff;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        padding-top: 10px;
        border-top: 1px solid #cccccc;
        font-size: 14px;
        color: #666666;
    }
    .stSlider label {
        font-weight: bold;
        color: #3366ff;
    }
    .info-box {
        background-color: #e6f2ff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
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
    # For now, we'll use a predetermined importance from your analysis
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

# Main application code
def main():
    # Header
    st.markdown('<div class="main-header">Diabetes Risk Prediction System</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4320/4320371.png", width=150)
    st.sidebar.markdown("## About This Tool")
    st.sidebar.info("""
    This application uses a K-Nearest Neighbors (KNN) machine learning model 
    trained on the Pima Indians Diabetes Dataset to predict diabetes risk. 
    
    Enter your health metrics to receive a personalized risk assessment.
    """)
    
    # Load the model
    model = load_model()
    
    if model:
        # Create two columns for input and visualization
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown('<div class="sub-header">Enter Your Health Information</div>', unsafe_allow_html=True)
            
            # Input fields with sliders and number inputs
            pregnancies = st.slider("Number of Pregnancies", 0, 17, 3, 1)
            
            glucose = st.slider("Glucose Level (mg/dL)", 50, 200, 120, 1)
            if not is_in_normal_range('Glucose', glucose):
                st.info("Normal fasting glucose range is 70-99 mg/dL")
            
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 40, 130, 70, 1)
            if not is_in_normal_range('BloodPressure', blood_pressure):
                st.info("Normal blood pressure is 90-120 mm Hg")
            
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20, 1)
            
            insulin = st.slider("Insulin Level (µU/ml)", 0, 846, 80, 1)
            if not is_in_normal_range('Insulin', insulin):
                st.info("Normal fasting insulin range is 16-166 µU/ml")
            
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
            st.info(f"BMI Status: {bmi_status}")
            
            diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
            
            age = st.slider("Age", 21, 90, 33, 1)
            
            # Create input data array
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, diabetes_pedigree, age]
            
            # Prediction button
            if st.button('Predict Diabetes Risk', key='predict_button'):
                prediction, probability = make_prediction(model, input_data)
                
                # Display prediction result
                st.markdown("### Prediction Result")
                
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
                        <div>Probability of diabetes: {probability:.2%}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Risk factors analysis
                st.markdown("### Risk Factor Analysis")
                
                # Check values outside normal range
                ranges = get_normal_ranges()
                features_outside_range = []
                
                for i, feature in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']):
                    min_val, max_val = ranges[feature]
                    if not (min_val <= input_data[i] <= max_val):
                        features_outside_range.append(feature)
                
                if features_outside_range:
                    st.markdown(f"The following values are outside the normal range:")
                    for feature in features_outside_range:
                        st.markdown(f"- **{feature}**")
                else:
                    st.markdown("All values are within normal ranges.")
        
        with col2:
            st.markdown('<div class="sub-header">Health Metrics Analysis</div>', unsafe_allow_html=True)
            
            # Create a radar chart for user's metrics
            st.markdown("### Your Health Metrics Profile")
            
            # Get normal ranges for scaling
            ranges = get_normal_ranges()
            
            # Prepare data for radar chart
            categories = ['Glucose', 'Blood Pressure', 'BMI', 
                         'Diabetes Pedigree', 'Age', 'Insulin']
            
            # Normalize values between 0 and 1 for radar chart
            values = [
                glucose / 200,
                blood_pressure / 130,
                bmi / 50,
                diabetes_pedigree / 2.5,
                age / 90,
                insulin / 400
            ]
            
            # Create the radar chart
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            
            # Number of variables
            N = len(categories)
            
            # Angle of each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Values to plot
            values += values[:1]  # Close the loop
            
            # Draw the chart
            ax.plot(angles, values, linewidth=2, linestyle='solid', color='#3366FF')
            ax.fill(angles, values, alpha=0.25, color='#3366FF')
            
            # Add labels
            plt.xticks(angles[:-1], categories)
            
            # Customize the chart
            ax.set_rlabel_position(0)
            plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=8)
            plt.ylim(0, 1)
            
            # Add a title
            plt.title("Health Metrics Profile", size=14, color='#3366FF', y=1.1)
            
            # Display the chart
            st.pyplot(fig)
            
            # Feature importance visualization
            st.markdown('<div class="feature-importance">', unsafe_allow_html=True)
            st.markdown("### Factors Affecting Diabetes Risk")
            
            importances = get_feature_importance()
            features = list(importances.keys())
            values = list(importances.values())
            
            # Create a horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort importances
            sorted_idx = np.argsort(values)
            ax.barh([features[i] for i in sorted_idx], [values[i] for i in sorted_idx], color='#3366FF')
            
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance for Diabetes Prediction')
            
            # Display the chart
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recommendations
            if 'prediction' in locals():
                st.markdown("### Recommendations")
                
                if prediction == 1:
                    st.markdown("""
                    Based on your metrics, here are some recommendations:
                    
                    1. **Consult a healthcare professional** for proper diagnosis and treatment
                    2. **Monitor your blood glucose levels** regularly
                    3. **Maintain a healthy diet** low in simple carbohydrates and sugars
                    4. **Exercise regularly** - aim for at least 150 minutes of moderate activity per week
                    5. **Maintain a healthy weight** to improve insulin sensitivity
                    """)
                else:
                    st.markdown("""
                    Based on your metrics, here are some recommendations to maintain good health:
                    
                    1. **Continue regular health check-ups** with your healthcare provider
                    2. **Maintain a balanced diet** rich in vegetables, fruits, and whole grains
                    3. **Stay physically active** with regular exercise
                    4. **Monitor your weight** and maintain a healthy BMI
                    5. **Get adequate sleep** and manage stress effectively
                    """)
    
    # Footer
    st.markdown('<div class="footer">Developed using Streamlit and K-Nearest Neighbors (KNN) Algorithm</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()