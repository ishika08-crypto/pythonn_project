# %%
import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# %%
# ── Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('calories.csv')
print("Shape     :", df.shape)
print("Columns   :", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nInfo:")
df.info()
print("\nDescription:")
print(df.describe())
# ── EDA Plot 1: Height vs Weight scatter ───────────────────────────────────
sns.scatterplot(x='Height', y='Weight', data=df)
plt.title("Height vs Weight")
plt.tight_layout()
plt.savefig("scatter_height_weight.png", dpi=120)
plt.show()

# Age vs Calories
plt.figure(figsize=(5,4))
sns.scatterplot(x='Age', y='Calories', data=df)
plt.title("Age vs Calories")
plt.show()

# Height vs Calories
plt.figure(figsize=(5,4))
sns.scatterplot(x='Height', y='Calories', data=df)
plt.title("Height vs Calories")
plt.show()


# %%
# Weight vs Calories
plt.figure(figsize=(5,4))
sns.scatterplot(x='Weight', y='Calories', data=df)
plt.title("Weight vs Calories")
plt.show()

# %%

# Duration vs Calories
plt.figure(figsize=(5,4))
sns.scatterplot(x='Duration', y='Calories', data=df)
plt.title("Duration vs Calories")
plt.show()

# %%
# Calories Distribution
sns.histplot(df["Calories"], kde=True)
plt.title("Calories Distribution")
plt.show()

# %%

# Height Distribution
sns.histplot(df["Height"], kde=True)
plt.title("Height Distribution")
plt.show()


# %%
# Weight Distribution
sns.histplot(df["Weight"], kde=True)
plt.title("Weight Distribution")
plt.show()

# %%
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# %%
# Convert Gender into Numbers
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

# %%
# Remove User_ID
df = df.drop("User_ID", axis=1)

# %%
# Features and Target
X = df.drop("Calories", axis=1)
y = df["Calories"]

# %%
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# %%
# ------------------ Train Multiple Models ------------------

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

results = []

print("\nModel Performance\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append([name, mae, r2])

    print(f"{name}")
    print(f"MAE : {mae:.2f}")
    print(f"R2  : {r2:.4f}")
    print("-"*30)

# %%
results_df = pd.DataFrame(results, columns=["Model", "MAE", "R2 Score"])

print(results_df)

# %%
#Select the Best Model
best_model = results_df.loc[results_df["R2 Score"].idxmax()]

print("Best Model")
print(best_model)

# %%
#Use the Best Model for Prediction
best = models[best_model["Model"]]

sample = pd.DataFrame({
    "Gender":[1],
    "Age":[25],
    "Height":[175],
    "Weight":[70],
    "Duration":[30],
    "Heart_Rate":[120],
    "Body_Temp":[40]
})

prediction = best.predict(sample)

print("Predicted Calories Burned:", prediction[0])





