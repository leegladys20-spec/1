import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .result-diabetic {
        background-color: #FFE5E5;
        border: 2px solid #C44E52;
    }
    .result-non-diabetic {
        background-color: #E5F5E5;
        border: 2px solid #4CAF50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .feature-importance {
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and preprocessors
@st.cache_resource
def load_models():
    try:
        with open('diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('imputer.pkl', 'rb') as f:
            imputer = pickle.load(f)
        return model, scaler, imputer
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        return None, None, None

# Load the models
model, scaler, imputer = load_models()

# Header
st.markdown('<p class="main-header">🩺 Diabetes Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter patient health metrics to predict diabetes risk</p>', unsafe_allow_html=True)

# Sidebar for patient information
with st.sidebar:
    st.header("📋 Patient Information")
    st.markdown("---")
    
    # Create input fields
    st.subheader("Demographics")
    age = st.slider("Age", 18, 100, 30, help="Patient's age in years")
    pregnancies = st.number_input("Number of Pregnancies", 0, 20, 2, help="Number of times pregnant")
    
    st.subheader("Clinical Measurements")
    glucose = st.number_input("Glucose Level (mg/dL)", 50, 200, 120, help="Plasma glucose concentration")
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", 40, 140, 70, help="Diastolic blood pressure")
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 60, 25, help="Triceps skin fold thickness")
    insulin = st.number_input("Insulin Level (μU/mL)", 0, 800, 80, help="2-Hour serum insulin")
    bmi = st.number_input("BMI (kg/m²)", 10, 60, 30.0, help="Body mass index")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01, help="Diabetes pedigree function")
    
    st.markdown("---")
    
    # Predict button
    predict_button = st.button("🔄 Predict Diabetes Risk", type="primary", use_container_width=True)

# Main content area
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("📊 Age", f"{age} years")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("⚖️ BMI", f"{bmi:.1f} kg/m²")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("🩸 Glucose", f"{glucose} mg/dL")
    st.markdown('</div>', unsafe_allow_html=True)

# Feature importance visualization
if model is not None:
    st.markdown("---")
    st.subheader("🔍 Model Feature Importance")
    
    # Get feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        importance = model.feature_importances_
        
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.argsort(importance)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importance)))
        ax.barh(range(len(indices)), importance[indices], color=colors)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance in Diabetes Prediction')
        
        for i, v in enumerate(importance[indices]):
            ax.text(v + 0.005, i, f'{v:.3f}', va='center')
        
        st.pyplot(fig)
        plt.close()

# Prediction logic
if predict_button and model is not None and scaler is not None and imputer is not None:
    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, diabetes_pedigree, age]])
    
    # Create DataFrame for imputation
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    # Impute missing values (handle zeros as missing)
    # Note: We replace zeros with NaN for imputation (same as training)
    cols_with_zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero_as_missing:
        if col in input_df.columns:
            input_df[col] = input_df[col].replace(0, np.nan)
    
    # Transform using the fitted imputer
    input_imputed = imputer.transform(input_df)
    
    # Predict
    prediction = model.predict(input_imputed)
    prediction_proba = model.predict_proba(input_imputed)
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Create columns for results
    col_result1, col_result2, col_result3 = st.columns(3)
    
    with col_result1:
        if prediction[0] == 1:
            st.markdown(f"""
                <div class="result-box result-diabetic">
                    <h2 style="color: #C44E52; margin:0;">⚠️ Diabetic</h2>
                    <p style="font-size:1.1rem; margin:0;">High risk of diabetes detected</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-box result-non-diabetic">
                    <h2 style="color: #4CAF50; margin:0;">✅ Non-Diabetic</h2>
                    <p style="font-size:1.1rem; margin:0;">Low risk of diabetes detected</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col_result2:
        st.metric("Prediction Confidence", f"{prediction_proba[0][prediction[0]]:.2%}")
        st.caption("Confidence level of the prediction")
    
    with col_result3:
        st.metric("Risk Score", f"{prediction_proba[0][1]:.2%}")
        st.caption("Probability of having diabetes")
    
    # Detailed probability breakdown
    st.markdown("---")
    st.subheader("📈 Probability Breakdown")
    
    # Create a horizontal bar chart for probabilities
    fig, ax = plt.subplots(figsize=(8, 4))
    prob_data = ['Non-Diabetic', 'Diabetic']
    prob_values = [prediction_proba[0][0], prediction_proba[0][1]]
    colors_probs = ['#4CAF50', '#C44E52']
    
    bars = ax.bar(prob_data, prob_values, color=colors_probs, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Diabetes Risk Probability')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    ax.legend()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom')
    
    st.pyplot(fig)
    plt.close()
    
    # Additional clinical insights
    st.markdown("---")
    st.subheader("💡 Clinical Insights")
    
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        st.markdown("**Risk Factors Present:**")
        risk_factors = []
        if glucose > 126:
            risk_factors.append("🟡 High Glucose (>126 mg/dL)")
        if bmi > 30:
            risk_factors.append("🟡 High BMI (>30 kg/m²)")
        if age > 45:
            risk_factors.append("🟡 Age > 45 years")
        if diabetes_pedigree > 0.8:
            risk_factors.append("🟡 High Diabetes Pedigree Function")
        if blood_pressure > 80:
            risk_factors.append("🟡 High Blood Pressure (>80 mm Hg)")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.write("✅ No major risk factors detected")
    
    with col_insight2:
        st.markdown("**Recommendations:**")
        if prediction[0] == 1:
            st.write("""
                • 🏥 Consult a healthcare provider immediately
                • 📊 Monitor blood glucose regularly
                • 🥗 Follow a healthy diet plan
                • 🏃 Regular exercise routine
                • 💊 Take prescribed medications
            """)
        else:
            st.write("""
                • ✅ Continue healthy lifestyle
                • 🥗 Maintain balanced diet
                • 🏃 Regular physical activity
                • 📈 Regular health check-ups
                • 🩺 Monitor symptoms
            """)
    
elif predict_button and model is None:
    st.error("❌ Model not loaded. Please train the model first.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>⚠️ This is a prediction tool for educational purposes only. Always consult a healthcare professional for medical advice.</p>
        <p>📊 Model Accuracy: ~78% | Built with Random Forest Classifier</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar information
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.info("""
    **Model Information:**
    - Algorithm: Random Forest
    - Accuracy: ~78%
    - AUC-ROC: ~0.83
    - Features: 8 clinical measurements
    
    **Data Source:**
    Pima Indians Diabetes Database
    
    **Note:**
    This model was trained on historical data and should be used as a screening tool, not a diagnostic tool.
    """)
    
    st.markdown("---")
    st.subheader("📚 Reference Ranges")
    st.write("""
    - **Glucose:** 70-99 mg/dL (Normal)
    - **BMI:** 18.5-24.9 (Normal)
    - **Blood Pressure:** <120/80 mm Hg (Normal)
    - **Age:** Risk increases with age
    """)
