# HEART DISEASE PREDICTION PROJECT

## PROJECT OVERVIEW:
This project builds a machine learning model to predict the risk of heart disease 
based on patient medical data. The final model is deployed as a Streamlit web 
application for real-time predictions.

## PROCESS STEPS:
STEP 1: DATA LOADING & SETUP
* Import required libraries: NumPy, Pandas, Seaborn, Matplotlib
* Load heart disease dataset from: Dataset/heart.csv
* Dataset contains 11 features and 1 target variable (HeartDisease)

STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
* Display dataset shape and columns
* Check data types and info()
* Generate descriptive statistics (mean, std, min, max)
* Check for duplicates (confirmed 0 duplicates)
* Check for missing values (confirmed 0 null values)
* Visualize target variable distribution (HeartDisease counts)
* Plot histograms for numerical features: Age, RestingBP, Cholesterol, MaxHR
* Create count plots for categorical variables by HeartDisease status
* Generate correlation heatmap for all numeric features
* Use boxplots and violin plots for distribution analysis
   
STEP 3: DATA CLEANING & HANDLING
* Identified 0 values in Cholesterol and RestingBP (likely missing data)
* Replaced 0 values in Cholesterol with mean (excluding zeros)
* Replaced 0 values in RestingBP with mean (excluding zeros)
* Rounded values to 2 decimal places for consistency

STEP 4: DATA PREPROCESSING
* Applied one-hot encoding using pd.get_dummies() with drop_first=True
* Converted all encoded features to integer type
* This created dummy variables for categorical features:
  * Sex (M/F)
  * ChestPainType (ATA, NAP, TA, ASY)
  * RestingECG (Normal, ST, LVH)
  * ExerciseAngina (Y/N)
  * ST_Slope (Up, Flat, Down)

STEP 5: FEATURE & TARGET SEPARATION
* Separated features (X) and target (y)
* X: All features except HeartDisease
* y: HeartDisease column (0 = Low Risk, 1 = High Risk)
   
STEP 6: TRAIN-TEST SPLIT
* Divided data into training and testing sets
* Test size: 33% (default random_state: 42)
* Training set: 67% of data
* Testing set: 33% of data
   
STEP 7: FEATURE SCALING
* Applied StandardScaler to normalize feature values
* Fit scaler on training data (X_train)
* Transformed both training and testing data
   
STEP 8: MODEL TRAINING & EVALUATION
* Trained 5 different machine learning models:
  a) Logistic Regression
  b) K-Nearest Neighbors (KNN)
  c) Naive Bayes
  d) Decision Trees
  e) Support Vector Machine (SVM with RBF Kernel)
  * Metrics used: Accuracy Score and F1 Score

STEP 9: MODEL SELECTION & DEPLOYMENT
* Selected KNN as the best performing model
* Saved artifacts as pickle files:
  * KNN_Heart.pkl - Trained KNN model
    * Scaler_Heart.pkl - StandardScaler object
    * Columns_Heart.pkl - Feature column names list
   
STEP 10: STREAMLIT WEB APPLICATION
* Created Heart_App.py using Streamlit framework
* Application collects user input:
  * Age (18-100 years)
  * Sex (M/F)
  * Chest Pain Type (ATA, NAP, TA, ASY)
  * Resting Blood Pressure (80-200 mm Hg)
  * Cholesterol (80-600 mg/dl)
  * Fasting Blood Sugar (0/1)
  * Resting ECG (Normal, ST, LVH)
  * Max Heart Rate (60-220 bpm)
  * Exercise-Induced Angina (Y/N)
  * Oldpeak/ST Depression (0.0-6.0)
  * ST Slope (Up, Flat, Down)
      
* Performs one-hot encoding matching training data format
* Scales input using saved scaler
* Makes prediction and displays result:
  * High Risk: ⚠️ (HeartDisease = 1)
  * Low Risk: ✅ (HeartDisease = 0)


## FILES GENERATED:
- Heart.ipynb - Main Jupyter notebook with complete analysis and training
- Heart_App.py - Streamlit web application for predictions
- KNN_Heart.pkl - Trained KNN model
- Scaler_Heart.pkl - StandardScaler for feature normalization
- Columns_Heart.pkl - Feature column names for prediction


## HOW TO USE THE APPLICATION:
1. Ensure all required packages are installed: streamlit, pandas, joblib, scikit-learn
2. Run the app using: streamlit run Heart_App.py
3. Enter patient medical information in the web interface
4. Click "Predict" button to get heart disease risk assessment
5. Result will show as either High Risk or Low Risk
