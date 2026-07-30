import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Calories Burnt Prediction",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Calories Burnt Prediction")
st.write("Predict calories burned using Machine Learning.")

# ----------------------------
# Load Dataset
# ----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("calories.csv")
    return df

df = load_data()

# ----------------------------
# Sidebar
# ----------------------------

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Dataset", "EDA", "Model Performance", "Prediction"]
)

# ----------------------------
# HOME
# ----------------------------

if menu == "Home":

    st.header("Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])

    with col2:
        st.metric("Missing Values", df.isnull().sum().sum())

    st.write(df.head())

# ----------------------------
# DATASET
# ----------------------------

elif menu == "Dataset":

    st.header("Dataset")

    st.dataframe(df)

    st.subheader("Statistics")

    st.write(df.describe())

# ----------------------------
# EDA
# ----------------------------

elif menu == "EDA":

    st.header("Exploratory Data Analysis")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=df,x="Height",y="Weight",ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=df,x="Age",y="Calories",ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=df,x="Height",y="Calories",ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=df,x="Weight",y="Calories",ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=df,x="Duration",y="Calories",ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df["Calories"],kde=True,ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.select_dtypes(include="number").corr(),
                annot=True,
                cmap="coolwarm",
                ax=ax)
    st.pyplot(fig)

# ----------------------------
# Train Model
# ----------------------------

encoder = LabelEncoder()

data = df.copy()

data["Gender"] = encoder.fit_transform(data["Gender"])

if "User_ID" in data.columns:
    data = data.drop("User_ID",axis=1)

X = data.drop("Calories",axis=1)
y = data["Calories"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Linear Regression":LinearRegression(),
    "Ridge Regression":Ridge(),
    "Lasso Regression":Lasso(),
    "Decision Tree":DecisionTreeRegressor(random_state=42),
    "Random Forest":RandomForestRegressor(random_state=42)
}

results=[]

best_model=None
best_score=-100

for name,model in models.items():

    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    mae=mean_absolute_error(y_test,pred)

    r2=r2_score(y_test,pred)

    results.append([name,mae,r2])

    if r2>best_score:
        best_score=r2
        best_model=model

results_df=pd.DataFrame(
    results,
    columns=["Model","MAE","R2 Score"]
)

# ----------------------------
# MODEL PERFORMANCE
# ----------------------------

if menu=="Model Performance":

    st.header("Model Comparison")

    st.dataframe(results_df)

    st.success(
        f"Best Model : {results_df.loc[results_df['R2 Score'].idxmax(),'Model']}"
    )

# ----------------------------
# PREDICTION
# ----------------------------

elif menu=="Prediction":

    st.header("Predict Calories Burned")

    gender=st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    age=st.number_input(
        "Age",
        10,
        80,
        25
    )

    height=st.number_input(
        "Height (cm)",
        100,
        220,
        170
    )

    weight=st.number_input(
        "Weight (kg)",
        20,
        200,
        70
    )

    duration=st.number_input(
        "Duration (minutes)",
        1,
        300,
        30
    )

    heart_rate=st.number_input(
        "Heart Rate",
        50,
        220,
        120
    )

    body_temp=st.number_input(
        "Body Temperature",
        35.0,
        45.0,
        40.0
    )

    if gender=="Male":
        gender=1
    else:
        gender=0

    sample=pd.DataFrame({
        "Gender":[gender],
        "Age":[age],
        "Height":[height],
        "Weight":[weight],
        "Duration":[duration],
        "Heart_Rate":[heart_rate],
        "Body_Temp":[body_temp]
    })

    if st.button("Predict Calories"):

        prediction=best_model.predict(sample)

        st.success(
            f"Estimated Calories Burned : {prediction[0]:.2f}"
        )