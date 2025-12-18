# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # Page configuration
# st.set_page_config(
#     page_title="Heart Disease Predictor",
#     page_icon="❤️",
#     layout="wide"
# )
#
# @st.cache_resource
# def load_all_models():
#     """Load all trained models and preprocessing objects"""
#     models = {}
#     errors = {}
#
#     model_files = {
#         'Random Forest': 'random_forest.pkl',
#         'Gradient Boosting': 'gradient_boosting.pkl',
#         'XGBoost': 'xgboost.pkl',
#         'KNN': 'knn.pkl',
#         'Voting Ensemble': 'voting_ensemble.pkl'
#     }
#
#     # Load individual models
#     for name, filename in model_files.items():
#         try:
#             with open(filename, 'rb') as f:
#                 models[name] = pickle.load(f)
#         except Exception as e:
#             models[name] = None
#             errors[name] = str(e)
#
#     # Load meta-ensemble
#     try:
#         with open('weighted_meta_ensemble.pkl', 'rb') as f:
#             meta_info = pickle.load(f)
#             models['Weighted Meta-Ensemble'] = meta_info
#     except Exception as e:
#         models['Weighted Meta-Ensemble'] = None
#         errors['Weighted Meta-Ensemble'] = str(e)
#
#     # Load preprocessing objects
#     try:
#         with open('scaler.pkl', 'rb') as f:
#             scaler = pickle.load(f)
#     except Exception as e:
#         st.error(f"Failed to load scaler: {e}")
#         scaler = None
#
#     try:
#         with open('label_encoders.pkl', 'rb') as f:
#             label_encoders = pickle.load(f)
#     except Exception as e:
#         label_encoders = {}
#
#     try:
#         with open('feature_names.pkl', 'rb') as f:
#             feature_names = pickle.load(f)
#     except Exception as e:
#         feature_names = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
#                          "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
#                          "Oldpeak", "ST_Slope"]
#
#     return models, errors, scaler, label_encoders, feature_names
#
#
# # Load everything
# models, model_errors, scaler, label_encoders, feature_names = load_all_models()
#
#
# # ============================================================================
# # HELPER FUNCTIONS
# # ============================================================================
#
# def preprocess_input(user_input, label_encoders, scaler):
#     """
#     Preprocess user input to match training data format
#
#     Args:
#         user_input: dict with feature values
#         label_encoders: dict of LabelEncoder objects
#         scaler: StandardScaler object
#
#     Returns:
#         Scaled feature array ready for prediction
#     """
#     # Create DataFrame with exact feature order
#     df = pd.DataFrame([user_input], columns=feature_names)
#
#     # Encode categorical variables
#     for col in df.columns:
#         if col in label_encoders:
#             # Transform categorical to numeric
#             try:
#                 df[col] = label_encoders[col].transform([df[col].iloc[0]])[0]
#             except:
#                 # Handle unknown categories
#                 df[col] = 0
#
#     # Ensure all values are numeric
#     df = df.astype(float)
#
#     # Scale features
#     if scaler is not None:
#         X_scaled = scaler.transform(df)
#     else:
#         X_scaled = df.values
#
#     return X_scaled
#
#
# def make_prediction(model, X_scaled, model_name=""):
#     """Make prediction with a single model"""
#     try:
#         prediction = model.predict(X_scaled)[0]
#
#         # Get probability if available
#         if hasattr(model, 'predict_proba'):
#             probability = model.predict_proba(X_scaled)[0]
#             prob_disease = probability[1]  # Probability of class 1 (disease)
#         else:
#             prob_disease = None
#
#         return prediction, prob_disease
#     except Exception as e:
#         st.error(f"Error in {model_name}: {e}")
#         return None, None
#
#
# def make_meta_prediction(meta_info, models, X_scaled):
#     """Make prediction using weighted meta-ensemble"""
#     try:
#         weights = meta_info['weights']
#         top_models = meta_info['top_models']
#
#         # Get predictions from top models
#         weighted_proba = 0
#         total_weight = 0
#
#         for model_name in top_models:
#             if model_name in models and models[model_name] is not None:
#                 weight = weights[model_name]
#
#                 if hasattr(models[model_name], 'predict_proba'):
#                     prob = models[model_name].predict_proba(X_scaled)[0][1]
#                 else:
#                     prob = models[model_name].predict(X_scaled)[0]
#
#                 weighted_proba += weight * prob
#                 total_weight += weight
#
#         # Normalize
#         if total_weight > 0:
#             weighted_proba /= total_weight
#
#         # Final prediction
#         prediction = 1 if weighted_proba > 0.5 else 0
#
#         return prediction, weighted_proba
#
#     except Exception as e:
#         st.error(f"Error in Meta-Ensemble: {e}")
#         return None, None
#
#
# # ============================================================================
# # USER INTERFACE
# # ============================================================================
#
# # Title and description
# st.title("❤️ Heart Disease Prediction System")
# st.markdown("""
# This application predicts the likelihood of heart disease based on patient health metrics.
# The models were trained on 11 key features using advanced machine learning algorithms.
# """)
#
# # Sidebar for model selection
# st.sidebar.header("⚙️ Model Selection")
# available_models = [name for name, model in models.items() if model is not None]
#
# if not available_models:
#     st.error("❌ No models loaded! Please ensure all .pkl files are in the same directory.")
#     st.stop()
#
# selected_model = st.sidebar.selectbox(
#     "Choose a model for prediction:",
#     available_models
# )
#
# st.sidebar.markdown("---")
# st.sidebar.info(f"""
# **Loaded Models:** {len(available_models)}/6
#
# **Features Used:** 11
# - Age, Sex, ChestPainType
# - RestingBP, Cholesterol
# - FastingBS, RestingECG
# - MaxHR, ExerciseAngina
# - Oldpeak, ST_Slope
# """)
#
# # Main tabs
# tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction", "📈 Model Info"])
#
# # ============================================================================
# # TAB 1: SINGLE PREDICTION
# # ============================================================================
#
# with tab1:
#     st.header("Enter Patient Information")
#
#     col1, col2, col3 = st.columns(3)
#
#     with col1:
#         st.subheader("Basic Information")
#         age = st.number_input("Age", min_value=1, max_value=120, value=50)
#
#         sex = st.selectbox("Sex", ["M", "F"])
#
#         chest_pain = st.selectbox(
#             "Chest Pain Type",
#             ["ATA", "NAP", "ASY", "TA"],
#             help="ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic, TA: Typical Angina"
#         )
#
#         resting_bp = st.number_input(
#             "Resting Blood Pressure (mm Hg)",
#             min_value=0,
#             max_value=300,
#             value=120
#         )
#
#     with col2:
#         st.subheader("Lab Results")
#         cholesterol = st.number_input(
#             "Cholesterol (mg/dl)",
#             min_value=0,
#             max_value=600,
#             value=200
#         )
#
#         fasting_bs = st.selectbox(
#             "Fasting Blood Sugar > 120 mg/dl",
#             [0, 1],
#             format_func=lambda x: "Yes" if x == 1 else "No"
#         )
#
#         resting_ecg = st.selectbox(
#             "Resting ECG",
#             ["Normal", "ST", "LVH"],
#             help="Normal, ST: ST-T wave abnormality, LVH: Left ventricular hypertrophy"
#         )
#
#         max_hr = st.number_input(
#             "Maximum Heart Rate",
#             min_value=60,
#             max_value=220,
#             value=150
#         )
#
#     with col3:
#         st.subheader("Exercise Test")
#         exercise_angina = st.selectbox(
#             "Exercise Induced Angina",
#             ["N", "Y"],
#             format_func=lambda x: "Yes" if x == "Y" else "No"
#         )
#
#         oldpeak = st.number_input(
#             "Oldpeak (ST Depression)",
#             min_value=-5.0,
#             max_value=10.0,
#             value=1.0,
#             step=0.1,
#             help="ST depression induced by exercise relative to rest"
#         )
#
#         st_slope = st.selectbox(
#             "ST Slope",
#             ["Up", "Flat", "Down"],
#             help="Slope of peak exercise ST segment"
#         )
#
#     # Create input dictionary
#     user_input = {
#         "Age": age,
#         "Sex": sex,
#         "ChestPainType": chest_pain,
#         "RestingBP": resting_bp,
#         "Cholesterol": cholesterol,
#         "FastingBS": fasting_bs,
#         "RestingECG": resting_ecg,
#         "MaxHR": max_hr,
#         "ExerciseAngina": exercise_angina,
#         "Oldpeak": oldpeak,
#         "ST_Slope": st_slope
#     }
#
#     # Predict button
#     if st.button("🔬 Predict", type="primary", use_container_width=True):
#         with st.spinner("Processing..."):
#             # Preprocess input
#             X_scaled = preprocess_input(user_input, label_encoders, scaler)
#
#             # Show input data
#             with st.expander("📋 View Input Data"):
#                 st.dataframe(pd.DataFrame([user_input]))
#
#             # Make prediction based on selected model
#             if selected_model == 'Weighted Meta-Ensemble':
#                 prediction, probability = make_meta_prediction(
#                     models[selected_model],
#                     models,
#                     X_scaled
#                 )
#             else:
#                 prediction, probability = make_prediction(
#                     models[selected_model],
#                     X_scaled,
#                     selected_model
#                 )
#
#             # Display results
#             st.markdown("---")
#             st.header("🎯 Prediction Results")
#
#             col_a, col_b = st.columns(2)
#
#             with col_a:
#                 if prediction == 1:
#                     st.error("⚠️ **POSITIVE** - Heart Disease Detected")
#                 else:
#                     st.success("✅ **NEGATIVE** - No Heart Disease Detected")
#
#             with col_b:
#                 if probability is not None:
#                     st.metric(
#                         "Confidence Level",
#                         f"{probability * 100:.1f}%",
#                         help="Probability of heart disease"
#                     )
#
#                     # Progress bar
#                     st.progress(probability)
#
#             # Risk assessment
#             st.markdown("---")
#             st.subheader("📊 Risk Assessment")
#
#             if probability is not None:
#                 if probability < 0.3:
#                     risk_level = "🟢 Low Risk"
#                     recommendation = "Maintain healthy lifestyle. Regular check-ups recommended."
#                 elif probability < 0.6:
#                     risk_level = "🟡 Moderate Risk"
#                     recommendation = "Consider consulting a cardiologist. Monitor symptoms closely."
#                 else:
#                     risk_level = "🔴 High Risk"
#                     recommendation = "Immediate medical consultation recommended. Seek professional advice."
#
#                 st.info(f"**Risk Level:** {risk_level}")
#                 st.warning(f"**Recommendation:** {recommendation}")
#
#             # Additional metrics from all models
#             with st.expander("🔍 Compare All Models"):
#                 results = []
#
#                 for model_name, model in models.items():
#                     if model is not None and model_name != 'Weighted Meta-Ensemble':
#                         pred, prob = make_prediction(model, X_scaled, model_name)
#                         if pred is not None:
#                             results.append({
#                                 'Model': model_name,
#                                 'Prediction': 'Disease' if pred == 1 else 'No Disease',
#                                 'Confidence': f"{prob * 100:.1f}%" if prob else "N/A"
#                             })
#
#                 if results:
#                     st.dataframe(pd.DataFrame(results), use_container_width=True)
#
# # ============================================================================
# # TAB 2: BATCH PREDICTION
# # ============================================================================
#
# with tab2:
#     st.header("📊 Batch Prediction from CSV")
#
#     st.info("""
#     Upload a CSV file with the following columns (in any order):
#
#     **Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope**
#     """)
#
#     # Sample CSV download
#     sample_data = {
#         "Age": [52, 45, 68],
#         "Sex": ["M", "F", "M"],
#         "ChestPainType": ["ATA", "NAP", "ASY"],
#         "RestingBP": [125, 130, 150],
#         "Cholesterol": [212, 204, 300],
#         "FastingBS": [0, 0, 1],
#         "RestingECG": ["Normal", "Normal", "ST"],
#         "MaxHR": [168, 156, 140],
#         "ExerciseAngina": ["N", "N", "Y"],
#         "Oldpeak": [1.0, 1.4, 2.3],
#         "ST_Slope": ["Up", "Flat", "Down"]
#     }
#
#     sample_df = pd.DataFrame(sample_data)
#     csv_sample = sample_df.to_csv(index=False).encode('utf-8')
#
#     st.download_button(
#         "⬇️ Download Sample CSV",
#         csv_sample,
#         "sample_heart_data.csv",
#         "text/csv",
#         key='download-csv'
#     )
#
#     # File upload
#     uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
#
#     if uploaded_file is not None:
#         try:
#             # Read CSV
#             df = pd.read_csv(uploaded_file)
#
#             st.success(f"✅ Loaded {len(df)} records")
#
#             # Verify columns
#             missing_cols = [col for col in feature_names if col not in df.columns]
#             if missing_cols:
#                 st.error(f"❌ Missing columns: {missing_cols}")
#             else:
#                 # Show preview
#                 st.subheader("📋 Data Preview")
#                 st.dataframe(df.head())
#
#                 if st.button("🔬 Run Batch Prediction", type="primary"):
#                     with st.spinner("Processing batch predictions..."):
#                         results_df = df.copy()
#
#                         # Process each row
#                         predictions = []
#                         probabilities = []
#
#                         for idx, row in df.iterrows():
#                             user_input = row.to_dict()
#                             X_scaled = preprocess_input(user_input, label_encoders, scaler)
#
#                             if selected_model == 'Weighted Meta-Ensemble':
#                                 pred, prob = make_meta_prediction(
#                                     models[selected_model],
#                                     models,
#                                     X_scaled
#                                 )
#                             else:
#                                 pred, prob = make_prediction(
#                                     models[selected_model],
#                                     X_scaled,
#                                     selected_model
#                                 )
#
#                             predictions.append(pred)
#                             probabilities.append(prob if prob is not None else 0.0)
#
#                         # Add results to dataframe
#                         results_df['Prediction'] = ['Disease' if p == 1 else 'No Disease' for p in predictions]
#                         results_df['Probability'] = [f"{p * 100:.1f}%" for p in probabilities]
#                         results_df['Risk_Level'] = [
#                             '🟢 Low' if p < 0.3 else '🟡 Moderate' if p < 0.6 else '🔴 High'
#                             for p in probabilities
#                         ]
#
#                         # Display results
#                         st.subheader("🎯 Prediction Results")
#                         st.dataframe(results_df, use_container_width=True)
#
#                         # Summary statistics
#                         col1, col2, col3 = st.columns(3)
#
#                         with col1:
#                             disease_count = sum(predictions)
#                             st.metric("Disease Detected", f"{disease_count}/{len(predictions)}")
#
#                         with col2:
#                             avg_prob = np.mean(probabilities)
#                             st.metric("Average Risk", f"{avg_prob * 100:.1f}%")
#
#                         with col3:
#                             high_risk = sum([1 for p in probabilities if p > 0.6])
#                             st.metric("High Risk Cases", f"{high_risk}")
#
#                         # Download results
#                         csv_results = results_df.to_csv(index=False).encode('utf-8')
#                         st.download_button(
#                             "⬇️ Download Results",
#                             csv_results,
#                             "heart_disease_predictions.csv",
#                             "text/csv"
#                         )
#
#         except Exception as e:
#             st.error(f"❌ Error processing file: {e}")
#
# # ============================================================================
# # TAB 3: MODEL INFORMATION
# # ============================================================================
#
# with tab3:
#     st.header("📈 Model Information")
#
#     # Model status
#     st.subheader("🤖 Loaded Models")
#
#     for model_name, model in models.items():
#         if model is not None:
#             st.success(f"✅ {model_name}")
#         else:
#             error_msg = model_errors.get(model_name, "Unknown error")
#             st.error(f"❌ {model_name}: {error_msg}")
#
#     # Feature information
#     st.subheader("📊 Feature Information")
#
#     st.markdown("""
#     The models use **11 features** for prediction:
#
#     1. **Age** - Patient age in years
#     2. **Sex** - M (Male) or F (Female)
#     3. **ChestPainType** - Type of chest pain (ATA, NAP, ASY, TA)
#     4. **RestingBP** - Resting blood pressure (mm Hg)
#     5. **Cholesterol** - Serum cholesterol (mg/dl)
#     6. **FastingBS** - Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
#     7. **RestingECG** - Resting electrocardiogram results (Normal, ST, LVH)
#     8. **MaxHR** - Maximum heart rate achieved
#     9. **ExerciseAngina** - Exercise-induced angina (Y = yes; N = no)
#     10. **Oldpeak** - ST depression induced by exercise
#     11. **ST_Slope** - Slope of peak exercise ST segment (Up, Flat, Down)
#     """)
#
#     # Model details
#     st.subheader("🔧 Model Architecture")
#
#     model_info = {
#         'Random Forest': 'Ensemble of 200 decision trees with max depth of 15',
#         'Gradient Boosting': 'Sequential ensemble with 200 estimators and learning rate 0.05',
#         'XGBoost': 'Optimized gradient boosting with 200 trees',
#         'KNN': 'K-Nearest Neighbors with k=5 and distance weighting',
#         'Voting Ensemble': 'Soft voting of Random Forest, Gradient Boosting, and XGBoost',
#         'Weighted Meta-Ensemble': 'Weighted combination of top 3 performing models'
#     }
#
#     for name, description in model_info.items():
#         if name in models and models[name] is not None:
#             with st.expander(f"ℹ️ {name}"):
#                 st.write(description)
#
#     # About
#     st.subheader("ℹ️ About")
#     st.info("""
#     **Heart Disease Prediction System**
#
#     This system uses machine learning to predict heart disease based on patient health metrics.
#     All models were trained on a comprehensive dataset and achieved over 85% accuracy.
#
#     **Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.
#     Always consult with healthcare professionals for medical decisions.
#     """)
#
# # Footer
# st.markdown("---")
# st.markdown("""
# <div style='text-align: center'>
#     <p>❤️ Heart Disease Prediction System | Powered by Machine Learning</p>
# </div>
# """, unsafe_allow_html=True)

#
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import warnings
# import plotly.graph_objects as go
# import plotly.express as px
#
# warnings.filterwarnings('ignore')
#
# # Page configuration
# st.set_page_config(
#     page_title="Heart Disease Predictor",
#     page_icon="❤️",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )
#
# # Custom CSS for professional healthcare styling
# st.markdown("""
# <style>
#     /* Import Google Fonts */
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
#
#     /* Global Styles */
#     * {
#         font-family: 'Inter', sans-serif;
#     }
#
#     /* Main container */
#     .main {
#         background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#         padding: 2rem;
#     }
#
#     /* Header styling */
#     h1 {
#         color: #1e3a8a;
#         font-weight: 700;
#         text-align: center;
#         padding: 1.5rem 0;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
#     }
#
#     h2, h3 {
#         color: #1e40af;
#         font-weight: 600;
#     }
#
#     /* Card-like containers */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 24px;
#         background-color: white;
#         padding: 1rem;
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
#
#     .stTabs [data-baseweb="tab"] {
#         height: 50px;
#         background-color: transparent;
#         border-radius: 8px;
#         color: #64748b;
#         font-weight: 600;
#         padding: 0 24px;
#         font-size: 16px;
#     }
#
#     .stTabs [aria-selected="true"] {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#     }
#
#     /* Input containers */
#     .stNumberInput, .stSelectbox {
#         background-color: white;
#         border-radius: 8px;
#         padding: 0.5rem;
#     }
#
#     /* Buttons */
#     .stButton > button {
#         width: 100%;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 0.75rem 2rem;
#         font-size: 16px;
#         font-weight: 600;
#         border-radius: 10px;
#         box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
#         transition: all 0.3s ease;
#     }
#
#     .stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
#     }
#
#     /* Info boxes */
#     .stAlert {
#         border-radius: 10px;
#         border-left: 5px solid;
#         padding: 1rem;
#         font-weight: 500;
#     }
#
#     /* Metric cards */
#     [data-testid="stMetricValue"] {
#         font-size: 2rem;
#         font-weight: 700;
#         color: #1e3a8a;
#     }
#
#     /* Column styling */
#     [data-testid="column"] {
#         background-color: white;
#         padding: 1.5rem;
#         border-radius: 12px;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.1);
#         margin: 0.5rem;
#     }
#
#     /* Dataframe styling */
#     .stDataFrame {
#         border-radius: 10px;
#         overflow: hidden;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
#
#     /* Progress bar */
#     .stProgress > div > div {
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         border-radius: 10px;
#     }
#
#     /* Expander */
#     .streamlit-expanderHeader {
#         background-color: white;
#         border-radius: 8px;
#         font-weight: 600;
#         color: #1e40af;
#     }
#
#     /* File uploader */
#     [data-testid="stFileUploader"] {
#         background-color: white;
#         border: 2px dashed #667eea;
#         border-radius: 10px;
#         padding: 2rem;
#     }
#
#     /* Download button */
#     .stDownloadButton > button {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         color: white;
#         border: none;
#         padding: 0.75rem 2rem;
#         font-size: 14px;
#         font-weight: 600;
#         border-radius: 8px;
#         transition: all 0.3s ease;
#     }
#
#     .stDownloadButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4);
#     }
#
#     /* Result cards */
#     .result-card {
#         background: white;
#         padding: 2rem;
#         border-radius: 12px;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.1);
#         margin: 1rem 0;
#         border-left: 5px solid #667eea;
#     }
#
#     /* Hide Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#
#     /* Custom spacing */
#     .block-container {
#         padding-top: 2rem;
#         padding-bottom: 2rem;
#     }
# </style>
# """, unsafe_allow_html=True)
#
#
# @st.cache_resource
# def load_all_models():
#     """Load all trained models and preprocessing objects"""
#     models = {}
#     errors = {}
#
#     model_files = {
#         'Random Forest': 'random_forest.pkl',
#         'Gradient Boosting': 'gradient_boosting.pkl',
#         'XGBoost': 'xgboost.pkl',
#         'KNN': 'knn.pkl',
#         'Voting Ensemble': 'voting_ensemble.pkl'
#     }
#
#     for name, filename in model_files.items():
#         try:
#             with open(filename, 'rb') as f:
#                 models[name] = pickle.load(f)
#         except Exception as e:
#             models[name] = None
#             errors[name] = str(e)
#
#     try:
#         with open('weighted_meta_ensemble.pkl', 'rb') as f:
#             meta_info = pickle.load(f)
#             models['Weighted Meta-Ensemble'] = meta_info
#     except Exception as e:
#         models['Weighted Meta-Ensemble'] = None
#         errors['Weighted Meta-Ensemble'] = str(e)
#
#     try:
#         with open('scaler.pkl', 'rb') as f:
#             scaler = pickle.load(f)
#     except Exception as e:
#         st.error(f"Failed to load scaler: {e}")
#         scaler = None
#
#     try:
#         with open('label_encoders.pkl', 'rb') as f:
#             label_encoders = pickle.load(f)
#     except Exception as e:
#         label_encoders = {}
#
#     try:
#         with open('feature_names.pkl', 'rb') as f:
#             feature_names = pickle.load(f)
#     except Exception as e:
#         feature_names = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
#                          "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
#                          "Oldpeak", "ST_Slope"]
#
#     return models, errors, scaler, label_encoders, feature_names
#
#
# models, model_errors, scaler, label_encoders, feature_names = load_all_models()
#
#
# def preprocess_input(user_input, label_encoders, scaler):
#     """Preprocess user input to match training data format"""
#     df = pd.DataFrame([user_input], columns=feature_names)
#
#     for col in df.columns:
#         if col in label_encoders:
#             try:
#                 df[col] = label_encoders[col].transform([df[col].iloc[0]])[0]
#             except:
#                 df[col] = 0
#
#     df = df.astype(float)
#
#     if scaler is not None:
#         X_scaled = scaler.transform(df)
#     else:
#         X_scaled = df.values
#
#     return X_scaled
#
#
# def make_prediction(model, X_scaled, model_name=""):
#     """Make prediction with a single model"""
#     try:
#         prediction = model.predict(X_scaled)[0]
#
#         if hasattr(model, 'predict_proba'):
#             probability = model.predict_proba(X_scaled)[0]
#             prob_disease = probability[1]
#         else:
#             prob_disease = None
#
#         return prediction, prob_disease
#     except Exception as e:
#         st.error(f"Error in {model_name}: {e}")
#         return None, None
#
#
# def make_meta_prediction(meta_info, models, X_scaled):
#     """Make prediction using weighted meta-ensemble"""
#     try:
#         weights = meta_info['weights']
#         top_models = meta_info['top_models']
#
#         weighted_proba = 0
#         total_weight = 0
#
#         for model_name in top_models:
#             if model_name in models and models[model_name] is not None:
#                 weight = weights[model_name]
#
#                 if hasattr(models[model_name], 'predict_proba'):
#                     prob = models[model_name].predict_proba(X_scaled)[0][1]
#                 else:
#                     prob = models[model_name].predict(X_scaled)[0]
#
#                 weighted_proba += weight * prob
#                 total_weight += weight
#
#         if total_weight > 0:
#             weighted_proba /= total_weight
#
#         prediction = 1 if weighted_proba > 0.5 else 0
#
#         return prediction, weighted_proba
#
#     except Exception as e:
#         st.error(f"Error in Meta-Ensemble: {e}")
#         return None, None
#
#
# # Title and description
# st.title("❤️ Heart Disease Prediction System")
# st.markdown("""
# <div style='text-align: center; padding: 1rem; background: white; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
#     <p style='font-size: 1.1rem; color: #64748b; margin: 0;'>
#         Advanced AI-powered diagnostic tool utilizing ensemble machine learning for accurate cardiovascular risk assessment.
#         <br><strong style='color: #dc2626;'>⚕️ For informational purposes only - Not a substitute for professional medical advice</strong>
#     </p>
# </div>
# """, unsafe_allow_html=True)
#
# # Main tabs
# tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📈 Model Performance"])
#
# # ============================================================================
# # TAB 1: SINGLE PREDICTION WITH ALL MODELS
# # ============================================================================
#
# with tab1:
#     st.markdown("<h2 style='text-align: center; color: #1e40af; margin-bottom: 2rem;'>Patient Health Assessment</h2>",
#                 unsafe_allow_html=True)
#
#     col1, col2, col3 = st.columns(3)
#
#     with col1:
#         st.markdown("### 👤 Demographics")
#         age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
#         sex = st.selectbox("Biological Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")
#
#         st.markdown("### 💓 Cardiovascular Metrics")
#         resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
#         max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
#
#     with col2:
#         st.markdown("### 🔬 Clinical Indicators")
#         chest_pain = st.selectbox(
#             "Chest Pain Type",
#             ["ATA", "NAP", "ASY", "TA"],
#             help="ATA: Atypical Angina | NAP: Non-Anginal Pain | ASY: Asymptomatic | TA: Typical Angina"
#         )
#         cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
#         fasting_bs = st.selectbox(
#             "Fasting Blood Sugar > 120 mg/dl",
#             [0, 1],
#             format_func=lambda x: "Yes (Diabetic)" if x == 1 else "No (Normal)"
#         )
#         resting_ecg = st.selectbox(
#             "Resting ECG Results",
#             ["Normal", "ST", "LVH"],
#             help="Normal | ST: ST-T Wave Abnormality | LVH: Left Ventricular Hypertrophy"
#         )
#
#     with col3:
#         st.markdown("### 🏃 Exercise Test Results")
#         exercise_angina = st.selectbox(
#             "Exercise-Induced Angina",
#             ["N", "Y"],
#             format_func=lambda x: "Yes" if x == "Y" else "No"
#         )
#         oldpeak = st.number_input(
#             "ST Depression (Oldpeak)",
#             min_value=-5.0,
#             max_value=10.0,
#             value=1.0,
#             step=0.1,
#             help="ST depression induced by exercise relative to rest"
#         )
#         st_slope = st.selectbox(
#             "ST Segment Slope",
#             ["Up", "Flat", "Down"],
#             help="Slope of peak exercise ST segment"
#         )
#
#     user_input = {
#         "Age": age,
#         "Sex": sex,
#         "ChestPainType": chest_pain,
#         "RestingBP": resting_bp,
#         "Cholesterol": cholesterol,
#         "FastingBS": fasting_bs,
#         "RestingECG": resting_ecg,
#         "MaxHR": max_hr,
#         "ExerciseAngina": exercise_angina,
#         "Oldpeak": oldpeak,
#         "ST_Slope": st_slope
#     }
#
#     st.markdown("<br>", unsafe_allow_html=True)
#
#     col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
#     with col_btn2:
#         predict_button = st.button("🔬 Analyze with All Models", type="primary", use_container_width=True)
#
#     if predict_button:
#         with st.spinner("🔄 Running comprehensive analysis across all models..."):
#             X_scaled = preprocess_input(user_input, label_encoders, scaler)
#
#             st.markdown("<br>", unsafe_allow_html=True)
#             st.markdown("<h2 style='text-align: center; color: #1e40af;'>🎯 Diagnostic Results</h2>",
#                         unsafe_allow_html=True)
#
#             # Collect all predictions
#             all_results = []
#             all_probabilities = []
#
#             for model_name, model in models.items():
#                 if model is not None:
#                     if model_name == 'Weighted Meta-Ensemble':
#                         prediction, probability = make_meta_prediction(model, models, X_scaled)
#                     else:
#                         prediction, probability = make_prediction(model, X_scaled, model_name)
#
#                     if prediction is not None:
#                         all_results.append({
#                             'Model': model_name,
#                             'Prediction': prediction,
#                             'Probability': probability if probability is not None else 0.5
#                         })
#                         if probability is not None:
#                             all_probabilities.append(probability)
#
#             # Consensus prediction
#             avg_probability = np.mean([r['Probability'] for r in all_results])
#             consensus_prediction = 1 if avg_probability > 0.5 else 0
#
#             # Main result display
#             st.markdown("<br>", unsafe_allow_html=True)
#             col_res1, col_res2 = st.columns([2, 1])
#
#             with col_res1:
#                 if consensus_prediction == 1:
#                     st.markdown("""
#                     <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
#                                 padding: 2rem; border-radius: 15px; border-left: 6px solid #dc2626;
#                                 box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2);'>
#                         <h2 style='color: #dc2626; margin: 0;'>⚠️ POSITIVE DETECTION</h2>
#                         <p style='font-size: 1.2rem; color: #991b1b; margin-top: 0.5rem;'>
#                             Heart disease indicators detected. Immediate medical consultation recommended.
#                         </p>
#                     </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.markdown("""
#                     <div style='background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
#                                 padding: 2rem; border-radius: 15px; border-left: 6px solid #16a34a;
#                                 box-shadow: 0 4px 12px rgba(22, 163, 74, 0.2);'>
#                         <h2 style='color: #16a34a; margin: 0;'>✅ NEGATIVE RESULT</h2>
#                         <p style='font-size: 1.2rem; color: #15803d; margin-top: 0.5rem;'>
#                             No significant heart disease indicators detected. Continue healthy lifestyle.
#                         </p>
#                     </div>
#                     """, unsafe_allow_html=True)
#
#             with col_res2:
#                 st.metric(
#                     "Overall Risk Score",
#                     f"{avg_probability * 100:.1f}%",
#                     help="Average probability across all models"
#                 )
#                 st.progress(avg_probability)
#
#                 if avg_probability < 0.3:
#                     risk_badge = "🟢 LOW RISK"
#                     risk_color = "#16a34a"
#                 elif avg_probability < 0.6:
#                     risk_badge = "🟡 MODERATE RISK"
#                     risk_color = "#eab308"
#                 else:
#                     risk_badge = "🔴 HIGH RISK"
#                     risk_color = "#dc2626"
#
#                 st.markdown(f"""
#                 <div style='background: white; padding: 1rem; border-radius: 10px;
#                             text-align: center; border: 3px solid {risk_color}; margin-top: 1rem;'>
#                     <h3 style='color: {risk_color}; margin: 0;'>{risk_badge}</h3>
#                 </div>
#                 """, unsafe_allow_html=True)
#
#             # Individual model results
#             st.markdown("<br><h3 style='color: #1e40af;'>📊 Individual Model Predictions</h3>", unsafe_allow_html=True)
#
#             # Create visualization
#             fig = go.Figure()
#
#             colors = ['#667eea' if r['Prediction'] == 0 else '#f5576c' for r in all_results]
#
#             fig.add_trace(go.Bar(
#                 x=[r['Probability'] * 100 for r in all_results],
#                 y=[r['Model'] for r in all_results],
#                 orientation='h',
#                 marker=dict(
#                     color=colors,
#                     line=dict(color='white', width=2)
#                 ),
#                 text=[f"{r['Probability'] * 100:.1f}%" for r in all_results],
#                 textposition='auto',
#                 hovertemplate='<b>%{y}</b><br>Probability: %{x:.1f}%<extra></extra>'
#             ))
#
#             fig.update_layout(
#                 title="Disease Probability by Model",
#                 xaxis_title="Probability of Heart Disease (%)",
#                 yaxis_title="",
#                 height=400,
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 font=dict(family="Inter, sans-serif", size=12),
#                 showlegend=False,
#                 xaxis=dict(range=[0, 100], gridcolor='#e5e7eb'),
#                 yaxis=dict(gridcolor='#e5e7eb')
#             )
#
#             fig.add_vline(x=50, line_dash="dash", line_color="red",
#                           annotation_text="Decision Threshold",
#                           annotation_position="top")
#
#             st.plotly_chart(fig, use_container_width=True)
#
#             # Detailed results table
#             results_df = pd.DataFrame(all_results)
#             results_df['Result'] = results_df['Prediction'].apply(lambda x: '🔴 Disease' if x == 1 else '🟢 No Disease')
#             results_df['Confidence'] = results_df['Probability'].apply(lambda x: f"{x * 100:.2f}%")
#
#             display_df = results_df[['Model', 'Result', 'Confidence']].copy()
#             st.dataframe(display_df, use_container_width=True, hide_index=True)
#
#             # Clinical recommendations
#             st.markdown("<br><h3 style='color: #1e40af;'>💡 Clinical Recommendations</h3>", unsafe_allow_html=True)
#
#             if avg_probability < 0.3:
#                 st.info("""
#                 **Low Risk Profile**
#                 - Continue regular health monitoring
#                 - Maintain balanced diet and regular exercise
#                 - Annual cardiovascular check-ups recommended
#                 - Monitor blood pressure and cholesterol levels
#                 """)
#             elif avg_probability < 0.6:
#                 st.warning("""
#                 **Moderate Risk Profile**
#                 - Schedule consultation with cardiologist within 2-4 weeks
#                 - Consider additional diagnostic tests (ECG, stress test, echocardiogram)
#                 - Implement lifestyle modifications (diet, exercise, stress management)
#                 - Monitor symptoms closely and seek immediate care if they worsen
#                 """)
#             else:
#                 st.error("""
#                 **High Risk Profile**
#                 - **URGENT**: Seek immediate medical evaluation
#                 - Comprehensive cardiac assessment required
#                 - Discuss treatment options with cardiovascular specialist
#                 - Consider emergency care if experiencing chest pain, shortness of breath, or other acute symptoms
#                 """)
#
# # ============================================================================
# # TAB 2: BATCH PREDICTION
# # ============================================================================
#
# with tab2:
#     st.markdown("<h2 style='text-align: center; color: #1e40af; margin-bottom: 2rem;'>Batch Patient Analysis</h2>",
#                 unsafe_allow_html=True)
#
#     st.markdown("""
#     <div style='background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
#                 padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;
#                 border-left: 5px solid #667eea;'>
#         <h4 style='color: #1e40af; margin-top: 0;'>📋 CSV File Requirements</h4>
#         <p style='color: #475569; margin-bottom: 0.5rem;'>
#             Upload a CSV file containing the following columns (order doesn't matter):
#         </p>
#         <code style='background: white; padding: 0.5rem; border-radius: 5px; display: block; color: #334155;'>
#             Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG,
#             MaxHR, ExerciseAngina, Oldpeak, ST_Slope
#         </code>
#     </div>
#     """, unsafe_allow_html=True)
#
#     col_sample1, col_sample2, col_sample3 = st.columns([1, 2, 1])
#
#     with col_sample2:
#         sample_data = {
#             "Age": [52, 45, 68],
#             "Sex": ["M", "F", "M"],
#             "ChestPainType": ["ATA", "NAP", "ASY"],
#             "RestingBP": [125, 130, 150],
#             "Cholesterol": [212, 204, 300],
#             "FastingBS": [0, 0, 1],
#             "RestingECG": ["Normal", "Normal", "ST"],
#             "MaxHR": [168, 156, 140],
#             "ExerciseAngina": ["N", "N", "Y"],
#             "Oldpeak": [1.0, 1.4, 2.3],
#             "ST_Slope": ["Up", "Flat", "Down"]
#         }
#         sample_df = pd.DataFrame(sample_data)
#         csv_sample = sample_df.to_csv(index=False).encode('utf-8')
#
#         st.download_button(
#             "⬇️ Download Sample CSV Template",
#             csv_sample,
#             "heart_disease_template.csv",
#             "text/csv",
#             use_container_width=True
#         )
#
#     st.markdown("<br>", unsafe_allow_html=True)
#     uploaded_file = st.file_uploader("📁 Upload Patient Data (CSV)", type=['csv'])
#
#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
#
#             st.success(f"✅ Successfully loaded {len(df)} patient records")
#
#             missing_cols = [col for col in feature_names if col not in df.columns]
#             if missing_cols:
#                 st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
#             else:
#                 st.markdown("### 📋 Data Preview")
#                 st.dataframe(df.head(10), use_container_width=True)
#
#                 col_analyze1, col_analyze2, col_analyze3 = st.columns([1, 2, 1])
#                 with col_analyze2:
#                     analyze_button = st.button("🔬 Analyze All Patients", type="primary", use_container_width=True)
#
#                 if analyze_button:
#                     with st.spinner(f"🔄 Processing {len(df)} patient records..."):
#                         results_df = df.copy()
#
#                         predictions = []
#                         probabilities = []
#
#                         progress_bar = st.progress(0)
#                         for idx, row in df.iterrows():
#                             user_input = row.to_dict()
#                             X_scaled = preprocess_input(user_input, label_encoders, scaler)
#
#                             # Use Weighted Meta-Ensemble for batch
#                             if 'Weighted Meta-Ensemble' in models and models['Weighted Meta-Ensemble'] is not None:
#                                 pred, prob = make_meta_prediction(
#                                     models['Weighted Meta-Ensemble'],
#                                     models,
#                                     X_scaled
#                                 )
#                             else:
#                                 # Fallback to first available model
#                                 available = [m for m in models.values() if m is not None][0]
#                                 pred, prob = make_prediction(available, X_scaled)
#
#                             predictions.append(pred)
#                             probabilities.append(prob if prob is not None else 0.5)
#                             progress_bar.progress((idx + 1) / len(df))
#
#                         progress_bar.empty()
#
#                         results_df['Prediction'] = ['Disease Detected' if p == 1 else 'No Disease' for p in predictions]
#                         results_df['Risk_Probability'] = [f"{p * 100:.1f}%" for p in probabilities]
#                         results_df['Risk_Level'] = [
#                             '🟢 Low' if p < 0.3 else '🟡 Moderate' if p < 0.6 else '🔴 High'
#                             for p in probabilities
#                         ]
#
#                         st.markdown("<br><h2 style='color: #1e40af;'>🎯 Analysis Results</h2>", unsafe_allow_html=True)
#
#                         # Summary metrics
#                         col1, col2, col3, col4 = st.columns(4)
#
#                         with col1:
#                             disease_count = sum(predictions)
#                             st.metric("Positive Cases", f"{disease_count}",
#                                       delta=f"{disease_count / len(predictions) * 100:.1f}%")
#
#                         with col2:
#                             no_disease = len(predictions) - disease_count
#                             st.metric("Negative Cases", f"{no_disease}",
#                                       delta=f"{no_disease / len(predictions) * 100:.1f}%")
#
#                         with col3:
#                             avg_prob = np.mean(probabilities)
#                             st.metric("Average Risk", f"{avg_prob * 100:.1f}%")
#
#                         with col4:
#                             high_risk = sum([1 for p in probabilities if p > 0.6])
#                             st.metric("High Risk Cases", f"{high_risk}",
#                                       delta=f"{high_risk / len(predictions) * 100:.1f}%")
#
#                         # Risk distribution chart
#                         st.markdown("### 📊 Risk Distribution")
#
#                         risk_counts = {
#                             'Low Risk': sum([1 for p in probabilities if p < 0.3]),
#                             'Moderate Risk': sum([1 for p in probabilities if 0.3 <= p < 0.6]),
#                             'High Risk': sum([1 for p in probabilities if p >= 0.6])
#                         }
#
#                         fig_pie = go.Figure(data=[go.Pie(
#                             labels=list(risk_counts.keys()),
#                             values=list(risk_counts.values()),
#                             marker=dict(colors=['#16a34a', '#eab308', '#dc2626']),
#                             hole=0.4,
#                             textinfo='label+percent',
#                             textfont=dict(size=14, family="Inter")
#                         )])
#
#                         fig_pie.update_layout(
#                             title="Patient Risk Classification",
#                             height=400,
#                             font=dict(family="Inter, sans-serif"),
#                             showlegend=True,
#                             paper_bgcolor='rgba(0,0,0,0)'
#                         )
#
#                         st.plotly_chart(fig_pie, use_container_width=True)
#
#                         # Detailed results table
#                         st.markdown("### 📄 Detailed Results")
#                         st.dataframe(results_df, use_container_width=True, height=400)
#
#                         # Download results
#                         csv_results = results_df.to_csv(index=False).encode('utf-8')
#                         col_down1, col_down2, col_down3 = st.columns([1, 2, 1])
#                         with col_down2:
#                             st.download_button(
#                                 "⬇️ Download Complete Analysis Report",
#                                 csv_results,
#                                 "heart_disease_analysis_results.csv",
#                                 "text/csv",
#                                 use_container_width=True
#                             )
#
#         except Exception as e:
#             st.error(f"❌ Error processing file: {e}")
#
# # ============================================================================
# # TAB 3: MODEL PERFORMANCE
# # ============================================================================
#
# with tab3:
#     st.markdown("<h2 style='text-align: center; color: #1e40af; margin-bottom: 2rem;'>Model Performance Analytics</h2>",
#                 unsafe_allow_html=True)
#
#     # Model performance data
#     performance_data = {
#         'Model': ['Random Forest', 'Weighted Meta-Ensemble', 'Voting Ensemble',
#                   'Gradient Boosting', 'XGBoost', 'KNN'],
#         'Accuracy': [97.78, 97.61, 97.44, 96.42, 95.73, 94.71],
#         'CV_Mean': [95.78, 97.61, 96.18, 95.27, 93.70, 94.32],
#         'CV_Std': [0.932, 0.000, 1.278, 0.542, 1.256, 0.738]
#     }
#
#     perf_df = pd.DataFrame(performance_data)
#
#     # Accuracy comparison chart
#     st.markdown("### 🎯 Model Accuracy Comparison")
#
#     fig_acc = go.Figure()
#
#     fig_acc.add_trace(go.Bar(
#         name='Test Accuracy',
#         x=perf_df['Model'],
#         y=perf_df['Accuracy'],
#         marker=dict(
#             color='#667eea',
#             line=dict(color='white', width=2)
#         ),
#         text=perf_df['Accuracy'].apply(lambda x: f"{x:.2f}%"),
#         textposition='outside'
#     ))
#
#     fig_acc.add_trace(go.Bar(
#         name='CV Mean Accuracy',
#         x=perf_df['Model'],
#         y=perf_df['CV_Mean'],
#         marker=dict(
#             color='#764ba2',
#             line=dict(color='white', width=2)
#         ),
#         text=perf_df['CV_Mean'].apply(lambda x: f"{x:.2f}%"),
#         textposition='outside'
#     ))
#
#     fig_acc.update_layout(
#         barmode='group',
#         xaxis_title="Model",
#         yaxis_title="Accuracy (%)",
#         height=500,
#         plot_bgcolor='rgba(0,0,0,0)',
#         paper_bgcolor='rgba(0,0,0,0)',
#         font=dict(family="Inter, sans-serif", size=12),
#         yaxis=dict(range=[90, 100], gridcolor='#e5e7eb'),
#         xaxis=dict(gridcolor='#e5e7eb'),
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1
#         )
#     )
#
#     st.plotly_chart(fig_acc, use_container_width=True)
#
#     # Cross-validation stability
#     col_cv1, col_cv2 = st.columns(2)
#
#     with col_cv1:
#         st.markdown("### 📊 Cross-Validation Stability")
#
#         fig_cv = go.Figure()
#
#         fig_cv.add_trace(go.Scatter(
#             x=perf_df['Model'],
#             y=perf_df['CV_Mean'],
#             mode='markers+lines',
#             name='CV Mean',
#             marker=dict(size=12, color='#667eea'),
#             line=dict(width=3, color='#667eea'),
#             error_y=dict(
#                 type='data',
#                 array=perf_df['CV_Std'],
#                 visible=True,
#                 color='#f5576c',
#                 thickness=2,
#                 width=6
#             )
#         ))
#
#         fig_cv.update_layout(
#             xaxis_title="",
#             yaxis_title="Accuracy (%) with Std Dev",
#             height=400,
#             plot_bgcolor='rgba(0,0,0,0)',
#             paper_bgcolor='rgba(0,0,0,0)',
#             font=dict(family="Inter, sans-serif", size=11),
#             yaxis=dict(range=[90, 100], gridcolor='#e5e7eb'),
#             xaxis=dict(tickangle=-45, gridcolor='#e5e7eb'),
#             showlegend=False
#         )
#
#         st.plotly_chart(fig_cv, use_container_width=True)
#
#     with col_cv2:
#         st.markdown("### 🔬 Performance Metrics Table")
#
#         display_perf = perf_df.copy()
#         display_perf['Accuracy'] = display_perf['Accuracy'].apply(lambda x: f"{x:.2f}%")
#         display_perf['CV_Mean'] = display_perf['CV_Mean'].apply(lambda x: f"{x:.2f}%")
#         display_perf['CV_Std'] = display_perf['CV_Std'].apply(lambda x: f"±{x:.3f}")
#
#         st.dataframe(display_perf, use_container_width=True, hide_index=True)
#
#     # Model comparison radar chart
#     st.markdown("### 🎪 Comprehensive Model Comparison")
#
#     fig_radar = go.Figure()
#
#     categories = ['Accuracy', 'CV Mean', 'Stability (inverse of std)']
#
#     for idx, row in perf_df.iterrows():
#         values = [
#             row['Accuracy'],
#             row['CV_Mean'],
#             100 - (row['CV_Std'] * 10)  # Convert std to stability score
#         ]
#         values.append(values[0])  # Close the radar chart
#
#         fig_radar.add_trace(go.Scatterpolar(
#             r=values,
#             theta=categories + [categories[0]],
#             name=row['Model'],
#             fill='toself',
#             opacity=0.6
#         ))
#
#     fig_radar.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[90, 100]
#             )
#         ),
#         height=600,
#         font=dict(family="Inter, sans-serif", size=12),
#         showlegend=True,
#         paper_bgcolor='rgba(0,0,0,0)'
#     )
#
#     st.plotly_chart(fig_radar, use_container_width=True)
#
#     # Model status
#     st.markdown("### 🤖 Model Loading Status")
#
#     status_cols = st.columns(3)
#
#     for idx, (model_name, model) in enumerate(models.items()):
#         with status_cols[idx % 3]:
#             if model is not None:
#                 st.success(f"✅ {model_name}")
#             else:
#                 error_msg = model_errors.get(model_name, "Unknown error")
#                 st.error(f"❌ {model_name}")
#                 st.caption(f"Error: {error_msg}")
#
#     # Feature importance (simulated)
#     st.markdown("### 📊 Feature Importance Analysis")
#
#     st.markdown("""
#     <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
#                 padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;
#                 border-left: 5px solid #f59e0b;'>
#         <p style='color: #78350f; margin: 0;'>
#             <strong>Key Predictive Features:</strong> Based on ensemble model analysis,
#             the most influential factors in heart disease prediction are ST_Slope,
#             Chest Pain Type, Exercise-Induced Angina, and Oldpeak values.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
#
#     # Technical information
#     st.markdown("### 🔧 Technical Specifications")
#
#     col_tech1, col_tech2 = st.columns(2)
#
#     with col_tech1:
#         st.markdown("""
#         **Model Architectures:**
#         - **Random Forest**: 200 trees, max depth 15
#         - **Gradient Boosting**: 200 estimators, learning rate 0.05
#         - **XGBoost**: 200 trees with optimized hyperparameters
#         - **KNN**: k=5 with distance weighting
#         """)
#
#     with col_tech2:
#         st.markdown("""
#         **Ensemble Methods:**
#         - **Voting Ensemble**: Soft voting (RF, GB, XGBoost)
#         - **Meta-Ensemble**: Weighted combination of top 3 models
#         - **Cross-Validation**: 5-fold stratified CV
#         - **Preprocessing**: StandardScaler normalization
#         """)
#
#     # Disclaimer
#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown("""
#     <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
#                 padding: 1.5rem; border-radius: 10px; border-left: 5px solid #dc2626;'>
#         <h4 style='color: #991b1b; margin-top: 0;'>⚕️ Medical Disclaimer</h4>
#         <p style='color: #7f1d1d; margin-bottom: 0;'>
#             This Heart Disease Prediction System is an <strong>educational and research tool</strong>
#             designed to demonstrate machine learning applications in healthcare. It should <strong>NOT</strong>
#             be used as a substitute for professional medical advice, diagnosis, or treatment.
#             <br><br>
#             <strong>Always consult qualified healthcare professionals</strong> for medical decisions
#             and diagnoses. The predictions made by this system are based on statistical models
#             and may not reflect individual medical complexities.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
#
# # Footer
# st.markdown("<br><br>", unsafe_allow_html=True)
# st.markdown("""
# <div style='text-align: center; padding: 2rem; background: white; border-radius: 10px;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
#     <h3 style='color: #1e40af; margin: 0;'>❤️ Heart Disease Prediction System</h3>
#     <p style='color: #64748b; margin: 0.5rem 0;'>
#         Powered by Advanced Machine Learning & AI
#     </p>
#     <p style='color: #94a3b8; font-size: 0.9rem; margin: 0;'>
#         Utilizing ensemble methods for accurate cardiovascular risk assessment
#     </p>
# </div>
# """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional healthcare styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem;
    }

    /* Header styling */
    h1 {
        color: #ffffff;
        font-weight: 700;
        text-align: center;
        padding: 1.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    h2, h3 {
        color: #e0e7ff;
        font-weight: 600;
    }

    /* Card-like containers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 0 24px;
        font-size: 16px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* Input containers */
    .stNumberInput, .stSelectbox {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* Input fields text color */
    input, select {
        color: #e2e8f0 !important;
        background-color: #0f172a !important;
    }

    /* Label text color */
    label {
        color: #cbd5e1 !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 16px;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
        padding: 1rem;
        font-weight: 500;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #a5b4fc;
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }

    /* Column styling */
    [data-testid="column"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        margin: 0.5rem;
        border: 1px solid #475569;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 8px;
        font-weight: 600;
        color: #e0e7ff;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 14px;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4);
    }

    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #1e293b;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Custom tooltip styling */
    .stTooltip {
        background: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }

    /* Custom footer */
    .custom-footer {
        text-align: center;
        padding: 0.8rem 1rem !important;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border-radius: 8px;
        margin-top: 2rem;
        border-top: 1px solid #334155;
        font-size: 0.85rem;
    }

    /* Animation for cards */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }

    /* Gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    """Load all trained models and preprocessing objects"""
    models = {}
    errors = {}

    model_files = {
        'Random Forest': 'random_forest.pkl',
        'Gradient Boosting': 'gradient_boosting.pkl',
        'XGBoost': 'xgboost.pkl',
        'KNN': 'knn.pkl',
        'Voting Ensemble': 'voting_ensemble.pkl'
    }

    for name, filename in model_files.items():
        try:
            with open(filename, 'rb') as f:
                models[name] = pickle.load(f)
        except Exception as e:
            models[name] = None
            errors[name] = str(e)

    try:
        with open('weighted_meta_ensemble.pkl', 'rb') as f:
            meta_info = pickle.load(f)
            models['Weighted Meta-Ensemble'] = meta_info
    except Exception as e:
        models['Weighted Meta-Ensemble'] = None
        errors['Weighted Meta-Ensemble'] = str(e)

    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load scaler: {e}")
        scaler = None

    try:
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
    except Exception as e:
        label_encoders = {}

    try:
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
    except Exception as e:
        feature_names = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
                         "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
                         "Oldpeak", "ST_Slope"]

    return models, errors, scaler, label_encoders, feature_names


models, model_errors, scaler, label_encoders, feature_names = load_all_models()


def preprocess_input(user_input, label_encoders, scaler):
    """Preprocess user input to match training data format"""
    df = pd.DataFrame([user_input], columns=feature_names)

    for col in df.columns:
        if col in label_encoders:
            try:
                df[col] = label_encoders[col].transform([df[col].iloc[0]])[0]
            except:
                df[col] = 0

    df = df.astype(float)

    if scaler is not None:
        X_scaled = scaler.transform(df)
    else:
        X_scaled = df.values

    return X_scaled


def make_prediction(model, X_scaled, model_name=""):
    """Make prediction with a single model"""
    try:
        prediction = model.predict(X_scaled)[0]

        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X_scaled)[0]
            prob_disease = probability[1]
        else:
            prob_disease = None

        return prediction, prob_disease
    except Exception as e:
        st.error(f"Error in {model_name}: {e}")
        return None, None


def make_meta_prediction(meta_info, models, X_scaled):
    """Make prediction using weighted meta-ensemble"""
    try:
        weights = meta_info['weights']
        top_models = meta_info['top_models']

        weighted_proba = 0
        total_weight = 0

        for model_name in top_models:
            if model_name in models and models[model_name] is not None:
                weight = weights[model_name]

                if hasattr(models[model_name], 'predict_proba'):
                    prob = models[model_name].predict_proba(X_scaled)[0][1]
                else:
                    prob = models[model_name].predict(X_scaled)[0]

                weighted_proba += weight * prob
                total_weight += weight

        if total_weight > 0:
            weighted_proba /= total_weight

        prediction = 1 if weighted_proba > 0.5 else 0

        return prediction, weighted_proba

    except Exception as e:
        st.error(f"Error in Meta-Ensemble: {e}")
        return None, None


# Title and description
st.title("❤️ Heart Disease Prediction System")
st.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
            border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.3); border: 1px solid #475569;'>
    <p style='font-size: 1.1rem; color: #cbd5e1; margin: 0;'>
        Advanced AI-powered diagnostic tool utilizing ensemble machine learning for accurate cardiovascular risk assessment.
        <br><strong style='color: #fca5a5;'>⚕️ For informational purposes only - Not a substitute for professional medical advice</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📈 Model Performance"])

# ============================================================================
# TAB 1: SINGLE PREDICTION WITH ALL MODELS
# ============================================================================

with tab1:
    st.markdown("<h2 style='text-align: center; color: #e0e7ff; margin-bottom: 2rem;'>Patient Health Assessment</h2>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👤 Demographics")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Biological Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")

        st.markdown("### 💓 Cardiovascular Metrics")
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
        max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)

    with col2:
        st.markdown("### 🔬 Clinical Indicators")
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "ASY", "TA"],
            help="ATA: Atypical Angina | NAP: Non-Anginal Pain | ASY: Asymptomatic | TA: Typical Angina"
        )
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1],
            format_func=lambda x: "Yes (Diabetic)" if x == 1 else "No (Normal)"
        )
        resting_ecg = st.selectbox(
            "Resting ECG Results",
            ["Normal", "ST", "LVH"],
            help="Normal | ST: ST-T Wave Abnormality | LVH: Left Ventricular Hypertrophy"
        )

    with col3:
        st.markdown("### 🏃 Exercise Test Results")
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["N", "Y"],
            format_func=lambda x: "Yes" if x == "Y" else "No"
        )
        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=-5.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="ST depression induced by exercise relative to rest"
        )
        st_slope = st.selectbox(
            "ST Segment Slope",
            ["Up", "Flat", "Down"],
            help="Slope of peak exercise ST segment"
        )

    user_input = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔬 Analyze with All Models", type="primary", use_container_width=True)

    if predict_button:
        with st.spinner("🔄 Running comprehensive analysis across all models..."):
            X_scaled = preprocess_input(user_input, label_encoders, scaler)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: #e0e7ff;'>🎯 Diagnostic Results</h2>",
                        unsafe_allow_html=True)

            # Collect all predictions
            all_results = []
            all_probabilities = []

            for model_name, model in models.items():
                if model is not None:
                    if model_name == 'Weighted Meta-Ensemble':
                        prediction, probability = make_meta_prediction(model, models, X_scaled)
                    else:
                        prediction, probability = make_prediction(model, X_scaled, model_name)

                    if prediction is not None:
                        all_results.append({
                            'Model': model_name,
                            'Prediction': prediction,
                            'Probability': probability if probability is not None else 0.5
                        })
                        if probability is not None:
                            all_probabilities.append(probability)

            # Consensus prediction
            avg_probability = np.mean([r['Probability'] for r in all_results])
            consensus_prediction = 1 if avg_probability > 0.5 else 0

            # Main result display
            st.markdown("<br>", unsafe_allow_html=True)
            col_res1, col_res2 = st.columns([2, 1])

            with col_res1:
                if consensus_prediction == 1:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                                padding: 2rem; border-radius: 15px; border-left: 6px solid #dc2626;
                                box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2);'>
                        <h2 style='color: #dc2626; margin: 0;'>⚠️ POSITIVE DETECTION</h2>
                        <p style='font-size: 1.2rem; color: #991b1b; margin-top: 0.5rem;'>
                            Heart disease indicators detected. Immediate medical consultation recommended.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                                padding: 2rem; border-radius: 15px; border-left: 6px solid #16a34a;
                                box-shadow: 0 4px 12px rgba(22, 163, 74, 0.2);'>
                        <h2 style='color: #16a34a; margin: 0;'>✅ NEGATIVE RESULT</h2>
                        <p style='font-size: 1.2rem; color: #15803d; margin-top: 0.5rem;'>
                            No significant heart disease indicators detected. Continue healthy lifestyle.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            with col_res2:
                st.metric(
                    "Overall Risk Score",
                    f"{avg_probability * 100:.1f}%",
                    help="Average probability across all models"
                )
                st.progress(avg_probability)

                if avg_probability < 0.3:
                    risk_badge = "🟢 LOW RISK"
                    risk_color = "#16a34a"
                elif avg_probability < 0.6:
                    risk_badge = "🟡 MODERATE RISK"
                    risk_color = "#eab308"
                else:
                    risk_badge = "🔴 HIGH RISK"
                    risk_color = "#dc2626"

                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 1rem; border-radius: 10px; 
                            text-align: center; border: 3px solid {risk_color}; margin-top: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.3);'>
                    <h3 style='color: {risk_color}; margin: 0;'>{risk_badge}</h3>
                </div>
                """, unsafe_allow_html=True)

            # Individual model results
            st.markdown("<br><h3 style='color: #e0e7ff;'>📊 Individual Model Predictions</h3>", unsafe_allow_html=True)

            # Create visualization
            fig = go.Figure()

            colors = ['#667eea' if r['Prediction'] == 0 else '#f5576c' for r in all_results]

            fig.add_trace(go.Bar(
                x=[r['Probability'] * 100 for r in all_results],
                y=[r['Model'] for r in all_results],
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=2)
                ),
                text=[f"{r['Probability'] * 100:.1f}%" for r in all_results],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Probability: %{x:.1f}%<extra></extra>'
            ))

            fig.update_layout(
                title="Disease Probability by Model",
                xaxis_title="Probability of Heart Disease (%)",
                yaxis_title="",
                height=400,
                plot_bgcolor='#1e293b',
                paper_bgcolor='#1e293b',
                font=dict(family="Inter, sans-serif", size=12, color='#e2e8f0'),
                showlegend=False,
                xaxis=dict(range=[0, 100], gridcolor='#475569'),
                yaxis=dict(gridcolor='#475569')
            )

            fig.add_vline(x=50, line_dash="dash", line_color="red",
                          annotation_text="Decision Threshold",
                          annotation_position="top")

            st.plotly_chart(fig, use_container_width=True)

            # Detailed results table
            results_df = pd.DataFrame(all_results)
            results_df['Result'] = results_df['Prediction'].apply(lambda x: '🔴 Disease' if x == 1 else '🟢 No Disease')
            results_df['Confidence'] = results_df['Probability'].apply(lambda x: f"{x * 100:.2f}%")

            display_df = results_df[['Model', 'Result', 'Confidence']].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Clinical recommendations
            st.markdown("<br><h3 style='color: #e0e7ff;'>💡 Clinical Recommendations</h3>", unsafe_allow_html=True)

            if avg_probability < 0.3:
                st.info("""
                **Low Risk Profile**
                - Continue regular health monitoring
                - Maintain balanced diet and regular exercise
                - Annual cardiovascular check-ups recommended
                - Monitor blood pressure and cholesterol levels
                """)
            elif avg_probability < 0.6:
                st.warning("""
                **Moderate Risk Profile**
                - Schedule consultation with cardiologist within 2-4 weeks
                - Consider additional diagnostic tests (ECG, stress test, echocardiogram)
                - Implement lifestyle modifications (diet, exercise, stress management)
                - Monitor symptoms closely and seek immediate care if they worsen
                """)
            else:
                st.error("""
                **High Risk Profile**
                - **URGENT**: Seek immediate medical evaluation
                - Comprehensive cardiac assessment required
                - Discuss treatment options with cardiovascular specialist
                - Consider emergency care if experiencing chest pain, shortness of breath, or other acute symptoms
                """)

# ============================================================================
# TAB 2: BATCH PREDICTION
# ============================================================================

with tab2:
    st.markdown("<h2 style='text-align: center; color: #e0e7ff; margin-bottom: 2rem;'>Batch Patient Analysis</h2>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style='background: linear-gradient(135deg, #312e81 0%, #4338ca 100%); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;
                border-left: 5px solid #667eea; box-shadow: 0 4px 12px rgba(0,0,0,0.3);'>
        <h4 style='color: #e0e7ff; margin-top: 0;'>📋 CSV File Requirements</h4>
        <p style='color: #cbd5e1; margin-bottom: 0.5rem;'>
            Upload a CSV file containing the following columns (order doesn't matter):
        </p>
        <code style='background: #1e293b; padding: 0.5rem; border-radius: 5px; display: block; color: #e2e8f0;'>
            Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, 
            MaxHR, ExerciseAngina, Oldpeak, ST_Slope
        </code>
    </div>
    """, unsafe_allow_html=True)

    col_sample1, col_sample2, col_sample3 = st.columns([1, 2, 1])

    with col_sample2:
        sample_data = {
            "Age": [52, 45, 68],
            "Sex": ["M", "F", "M"],
            "ChestPainType": ["ATA", "NAP", "ASY"],
            "RestingBP": [125, 130, 150],
            "Cholesterol": [212, 204, 300],
            "FastingBS": [0, 0, 1],
            "RestingECG": ["Normal", "Normal", "ST"],
            "MaxHR": [168, 156, 140],
            "ExerciseAngina": ["N", "N", "Y"],
            "Oldpeak": [1.0, 1.4, 2.3],
            "ST_Slope": ["Up", "Flat", "Down"]
        }
        sample_df = pd.DataFrame(sample_data)
        csv_sample = sample_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "⬇️ Download Sample CSV Template",
            csv_sample,
            "heart_disease_template.csv",
            "text/csv",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📁 Upload Patient Data (CSV)", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ Successfully loaded {len(df)} patient records")

            missing_cols = [col for col in feature_names if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                st.markdown("### 📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                col_analyze1, col_analyze2, col_analyze3 = st.columns([1, 2, 1])
                with col_analyze2:
                    analyze_button = st.button("🔬 Analyze All Patients", type="primary", use_container_width=True)

                if analyze_button:
                    with st.spinner(f"🔄 Processing {len(df)} patient records..."):
                        results_df = df.copy()

                        predictions = []
                        probabilities = []

                        progress_bar = st.progress(0)
                        for idx, row in df.iterrows():
                            user_input = row.to_dict()
                            X_scaled = preprocess_input(user_input, label_encoders, scaler)

                            # Use Weighted Meta-Ensemble for batch
                            if 'Weighted Meta-Ensemble' in models and models['Weighted Meta-Ensemble'] is not None:
                                pred, prob = make_meta_prediction(
                                    models['Weighted Meta-Ensemble'],
                                    models,
                                    X_scaled
                                )
                            else:
                                # Fallback to first available model
                                available = [m for m in models.values() if m is not None][0]
                                pred, prob = make_prediction(available, X_scaled)

                            predictions.append(pred)
                            probabilities.append(prob if prob is not None else 0.5)
                            progress_bar.progress((idx + 1) / len(df))

                        progress_bar.empty()

                        results_df['Prediction'] = ['Disease Detected' if p == 1 else 'No Disease' for p in predictions]
                        results_df['Risk_Probability'] = [f"{p * 100:.1f}%" for p in probabilities]
                        results_df['Risk_Level'] = [
                            '🟢 Low' if p < 0.3 else '🟡 Moderate' if p < 0.6 else '🔴 High'
                            for p in probabilities
                        ]

                        st.markdown("<br><h2 style='color: #e0e7ff;'>🎯 Analysis Results</h2>", unsafe_allow_html=True)

                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            disease_count = sum(predictions)
                            st.metric("Positive Cases", f"{disease_count}",
                                      delta=f"{disease_count / len(predictions) * 100:.1f}%")

                        with col2:
                            no_disease = len(predictions) - disease_count
                            st.metric("Negative Cases", f"{no_disease}",
                                      delta=f"{no_disease / len(predictions) * 100:.1f}%")

                        with col3:
                            avg_prob = np.mean(probabilities)
                            st.metric("Average Risk", f"{avg_prob * 100:.1f}%")

                        with col4:
                            high_risk = sum([1 for p in probabilities if p > 0.6])
                            st.metric("High Risk Cases", f"{high_risk}",
                                      delta=f"{high_risk / len(predictions) * 100:.1f}%")

                        # Risk distribution chart
                        st.markdown("### 📊 Risk Distribution")

                        risk_counts = {
                            'Low Risk': sum([1 for p in probabilities if p < 0.3]),
                            'Moderate Risk': sum([1 for p in probabilities if 0.3 <= p < 0.6]),
                            'High Risk': sum([1 for p in probabilities if p >= 0.6])
                        }

                        fig_pie = go.Figure(data=[go.Pie(
                            labels=list(risk_counts.keys()),
                            values=list(risk_counts.values()),
                            marker=dict(colors=['#16a34a', '#eab308', '#dc2626']),
                            hole=0.4,
                            textinfo='label+percent',
                            textfont=dict(size=14, family="Inter")
                        )])

                        fig_pie.update_layout(
                            title="Patient Risk Classification",
                            height=400,
                            font=dict(family="Inter, sans-serif", color='#e2e8f0'),
                            showlegend=True,
                            paper_bgcolor='#1e293b',
                            plot_bgcolor='#1e293b'
                        )

                        st.plotly_chart(fig_pie, use_container_width=True)

                        # Detailed results table
                        st.markdown("### 📄 Detailed Results")
                        st.dataframe(results_df, use_container_width=True, height=400)

                        # Download results
                        csv_results = results_df.to_csv(index=False).encode('utf-8')
                        col_down1, col_down2, col_down3 = st.columns([1, 2, 1])
                        with col_down2:
                            st.download_button(
                                "⬇️ Download Complete Analysis Report",
                                csv_results,
                                "heart_disease_analysis_results.csv",
                                "text/csv",
                                use_container_width=True
                            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# ============================================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================================

with tab3:
    st.markdown("<h2 style='text-align: center; color: #e0e7ff; margin-bottom: 2rem;'>Model Performance Analytics</h2>",
                unsafe_allow_html=True)

    # Model performance data
    performance_data = {
        'Model': ['Random Forest', 'Weighted Meta-Ensemble', 'Voting Ensemble',
                  'Gradient Boosting', 'XGBoost', 'KNN'],
        'Accuracy': [97.78, 97.61, 97.44, 96.42, 95.73, 94.71],
        'CV_Mean': [95.78, 97.61, 96.18, 95.27, 93.70, 94.32],
        'CV_Std': [0.932, 0.000, 1.278, 0.542, 1.256, 0.738]
    }

    perf_df = pd.DataFrame(performance_data)

    # Accuracy comparison chart
    st.markdown("### 🎯 Model Accuracy Comparison")

    fig_acc = go.Figure()

    fig_acc.add_trace(go.Bar(
        name='Test Accuracy',
        x=perf_df['Model'],
        y=perf_df['Accuracy'],
        marker=dict(
            color='#667eea',
            line=dict(color='white', width=2)
        ),
        text=perf_df['Accuracy'].apply(lambda x: f"{x:.2f}%"),
        textposition='outside'
    ))

    fig_acc.add_trace(go.Bar(
        name='CV Mean Accuracy',
        x=perf_df['Model'],
        y=perf_df['CV_Mean'],
        marker=dict(
            color='#764ba2',
            line=dict(color='white', width=2)
        ),
        text=perf_df['CV_Mean'].apply(lambda x: f"{x:.2f}%"),
        textposition='outside'
    ))

    fig_acc.update_layout(
        barmode='group',
        xaxis_title="Model",
        yaxis_title="Accuracy (%)",
        height=500,
        plot_bgcolor='#1e293b',
        paper_bgcolor='#1e293b',
        font=dict(family="Inter, sans-serif", size=12, color='#e2e8f0'),
        yaxis=dict(range=[90, 100], gridcolor='#475569'),
        xaxis=dict(gridcolor='#475569'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig_acc, use_container_width=True)

    # Cross-validation stability
    col_cv1, col_cv2 = st.columns(2)

    with col_cv1:
        st.markdown("### 📊 Cross-Validation Stability")

        fig_cv = go.Figure()

        fig_cv.add_trace(go.Scatter(
            x=perf_df['Model'],
            y=perf_df['CV_Mean'],
            mode='markers+lines',
            name='CV Mean',
            marker=dict(size=12, color='#667eea'),
            line=dict(width=3, color='#667eea'),
            error_y=dict(
                type='data',
                array=perf_df['CV_Std'],
                visible=True,
                color='#f5576c',
                thickness=2,
                width=6
            )
        ))

        fig_cv.update_layout(
            xaxis_title="",
            yaxis_title="Accuracy (%) with Std Dev",
            height=400,
            plot_bgcolor='#1e293b',
            paper_bgcolor='#1e293b',
            font=dict(family="Inter, sans-serif", size=11, color='#e2e8f0'),
            yaxis=dict(range=[90, 100], gridcolor='#475569'),
            xaxis=dict(tickangle=-45, gridcolor='#475569'),
            showlegend=False
        )

        st.plotly_chart(fig_cv, use_container_width=True)

    with col_cv2:
        st.markdown("### 🔬 Performance Metrics Table")

        display_perf = perf_df.copy()
        display_perf['Accuracy'] = display_perf['Accuracy'].apply(lambda x: f"{x:.2f}%")
        display_perf['CV_Mean'] = display_perf['CV_Mean'].apply(lambda x: f"{x:.2f}%")
        display_perf['CV_Std'] = display_perf['CV_Std'].apply(lambda x: f"±{x:.3f}")

        st.dataframe(display_perf, use_container_width=True, hide_index=True)

    # Model comparison radar chart
    st.markdown("### 🎪 Comprehensive Model Comparison")

    fig_radar = go.Figure()

    categories = ['Accuracy', 'CV Mean', 'Stability (inverse of std)']

    for idx, row in perf_df.iterrows():
        values = [
            row['Accuracy'],
            row['CV_Mean'],
            100 - (row['CV_Std'] * 10)  # Convert std to stability score
        ]
        values.append(values[0])  # Close the radar chart

        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=row['Model'],
            fill='toself',
            opacity=0.6
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[90, 100],
                gridcolor='#475569',
                color='#cbd5e1'
            ),
            angularaxis=dict(
                gridcolor='#475569',
                color='#cbd5e1'
            ),
            bgcolor='#1e293b'
        ),
        height=600,
        font=dict(family="Inter, sans-serif", size=12, color='#e2e8f0'),
        showlegend=True,
        paper_bgcolor='#1e293b'
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    # Model status
    st.markdown("### 🤖 Model Loading Status")

    status_cols = st.columns(3)

    for idx, (model_name, model) in enumerate(models.items()):
        with status_cols[idx % 3]:
            if model is not None:
                st.success(f"✅ {model_name}")
            else:
                error_msg = model_errors.get(model_name, "Unknown error")
                st.error(f"❌ {model_name}")
                st.caption(f"Error: {error_msg}")

    # Feature importance (simulated)
    st.markdown("### 📊 Feature Importance Analysis")

    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;
                border-left: 5px solid #667eea;'>
        <p style='color: #e2e8f0; margin: 0;'>
            <strong style='color: #a5b4fc;'>🔑 Key Predictive Features:</strong> 
            Based on ensemble model analysis, the most influential factors in heart disease prediction are:
            <br><br>
            <span style='color: #fca5a5;'>• ST_Slope</span> - Slope of ST segment during peak exercise<br>
            <span style='color: #fca5a5;'>• Chest Pain Type</span> - Type of chest discomfort experienced<br>
            <span style='color: #fca5a5;'>• Exercise-Induced Angina</span> - Chest pain during physical activity<br>
            <span style='color: #fca5a5;'>• Oldpeak</span> - ST depression value<br>
            <span style='color: #fca5a5;'>• Maximum Heart Rate</span> - Highest achieved heart rate
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Technical information
    st.markdown("### 🔧 Technical Specifications")

    col_tech1, col_tech2 = st.columns(2)

    with col_tech1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                    padding: 1.5rem; border-radius: 10px;'>
        **🏗️ Model Architectures:**
        - **Random Forest**: 200 trees, max depth 15
        - **Gradient Boosting**: 200 estimators, learning rate 0.05
        - **XGBoost**: 200 trees with optimized hyperparameters
        - **KNN**: k=5 with distance weighting
        </div>
        """, unsafe_allow_html=True)

    with col_tech2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                    padding: 1.5rem; border-radius: 10px;'>
        **🤝 Ensemble Methods:**
        - **Voting Ensemble**: Soft voting (RF, GB, XGBoost)
        - **Meta-Ensemble**: Weighted combination of top 3 models
        - **Cross-Validation**: 5-fold stratified CV
        - **Preprocessing**: StandardScaler normalization
        </div>
        """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: linear-gradient(135deg, #312e81 0%, #1e1b4b 100%); 
                padding: 1.5rem; border-radius: 10px; border-left: 5px solid #dc2626;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);'>
        <h4 style='color: #fca5a5; margin-top: 0;'>⚕️ Medical Disclaimer</h4>
        <p style='color: #e2e8f0; margin-bottom: 0;'>
            This Heart Disease Prediction System is an <strong>educational and research tool</strong> 
            designed to demonstrate machine learning applications in healthcare. It should <strong>NOT</strong> 
            be used as a substitute for professional medical advice, diagnosis, or treatment. 
            <br><br>
            <strong style='color: #fca5a5;'>Always consult qualified healthcare professionals</strong> for medical decisions 
            and diagnoses. The predictions made by this system are based on statistical models 
            and may not reflect individual medical complexities.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Custom Footer
st.markdown("""
<div class="custom-footer fade-in">
    <p style="color: #94a3b8; margin: 0; padding: 0;">
        Developed with ❤️ by 
        <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    -webkit-background-clip: text; 
                    -webkit-text-fill-color: transparent;
                    font-weight: 600;">
            Manvendra Yadav
        </span>
    </p>
    <p style="color: #64748b; margin: 0.3rem 0 0 0; padding: 0; font-size: 0.8rem;">
        Heart Disease Prediction System • Powered by Ensemble Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)