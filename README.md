Calories-Burnt-Prediction-Project

Machine Learning based Calories Burnt Prediction system using regression modeling, exploratory data analysis, and feature engineering techniques to estimate calories burned during physical activity.

🔥 Building a Regression-Based Calories Prediction System in Python

AIM: DATA SCIENCE & FITNESS ANALYTICS

In the modern health-tech landscape, accurately estimating calorie expenditure is critical for fitness tracking apps, wearables, and personalized health plans. This project implements a regression-based approach to predicting calories burnt using Python and Scikit-learn.

Python Pandas Scikit-learn Machine Learning License

📋 Table of Contents
Introduction
Dataset Description
Project Workflow
Exploratory Data Analysis
Model Building
Analytical Summary
How to Run
Conclusion
Introduction

This project explores how raw exercise and physiological data can be transformed into an accurate calorie-prediction tool using machine learning. We analyze key factors — such as duration, heart rate, body temperature, and demographic attributes — and use them to train regression models that estimate calories burnt per session.

Core Features:

Data cleaning and preprocessing pipeline
Exploratory Data Analysis (EDA) with visualizations
Feature correlation and importance analysis
Multiple regression model training and comparison
Performance evaluation using standard error metrics
Dataset Description

The project uses two merged datasets — calories.csv and exercise.csv — containing user demographic and exercise session data.

Column	Description
User_ID	Unique identifier for each user
Gender	Male / Female
Age	Age of the user (years)
Height	Height in cm
Weight	Weight in kg
Duration	Duration of exercise (minutes)
Heart_Rate	Average heart rate during exercise
Body_Temp	Body temperature during exercise (°C)
Calories	Target variable — calories burnt
Setup Code
python
import pandas as pd

# Load datasets
calories = pd.read_csv('calories.csv')
exercise = pd.read_csv('exercise.csv')

# Merge on User_ID
df = exercise.merge(calories, on='User_ID')
df.drop(columns='User_ID', inplace=True)
Project Workflow
Step 1: Data Preprocessing

Handling missing values, encoding categorical variables (Gender), and checking for outliers.

python
df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
df.isnull().sum()
Step 2: Train-Test Split
python
from sklearn.model_selection import train_test_split

X = df.drop(columns='Calories')
y = df['Calories']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
Exploratory Data Analysis

Correlation analysis reveals that Duration, Heart_Rate, and Body_Temp have the strongest positive relationship with calories burnt, while Age, Height, and Weight contribute secondary predictive value.

python
import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

Key Observations:

Duration and Heart_Rate show high positive correlation with Calories
Body_Temp rises consistently with longer exercise duration
Gender shows minimal direct correlation with calories burnt
Model Building

Several regression algorithms were trained and compared to identify the best-performing model.

python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"{name} -> MAE: {mean_absolute_error(y_test, preds):.2f}, R2: {r2_score(y_test, preds):.4f}")

Using ensemble models like Random Forest and XGBoost allows the system to capture non-linear relationships between physiological features and calorie expenditure that simple linear models miss.

Analytical Summary & Findings
Model	MAE	R² Score	Performance
Linear Regression	~8.5	~0.96	🟡 Good
Random Forest	~2.1	~0.99	🟢 Excellent
XGBoost	~1.8	~0.99	🟢 Excellent

Key Takeaways:

Best Model: XGBoost Regressor delivered the lowest error and highest R² score, making it the recommended model for deployment.
Feature Importance: Duration and Heart_Rate emerged as the most influential predictors of calories burnt.
Scalability: The model can be extended with additional features like step count, sleep data, or activity type for improved accuracy.
Optimization: Hyperparameter tuning (GridSearchCV) further improves XGBoost performance.
🚀 How to Run
Clone the repo
bash
git clone https://github.com/yourusername/calories-burnt-prediction.git
Install dependencies
bash
pip install -r requirements.txt
Run the training script
bash
python train_model.py
Make predictions
bash
python predict.py --age 25 --height 170 --weight 65 --duration 30 --heart_rate 105 --body_temp 40.1
Conclusion

By implementing this machine learning pipeline, we transform raw exercise and physiological data into an accurate, real-time calorie prediction tool. This system can be integrated into fitness apps, wearable devices, or health dashboards to give users instant, personalized feedback.

This is a foundational layer — it can be extended with deep learning models, real-time wearable data streaming, and deployment via Flask/Streamlit web apps.
