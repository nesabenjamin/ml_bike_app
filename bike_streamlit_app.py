import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
from sklearn.metrics import mean_squared_error

# ----------------------------
# Load your cleaned dataset
# ----------------------------
df_cleaned = pd.read_csv("df_cleaned.csv")   # <-- replace with your file
df_ols = df_cleaned.drop(['power','model_name'], axis=1)

# One-hot encode categorical columns
df_ols = pd.get_dummies(df_ols, columns=['brand','owner','location'], drop_first=True)

# Train/Test Setup
X = df_ols.drop(columns=['price']).astype(float)
# y = np.log1p(df_ols['price'])   # log(price)
y = df_ols['price']
X_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_const).fit()


# RMSE on training data
y_pred = ols_model.predict(X_const)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# Streamlit UI
# ----------------------------

st.title("🏍️ Know the Value")
st.markdown("Predict used bike prices using OLS regression.")
st.sidebar.header("Enter Bike Details")

# --- User Inputs ---
model_year = st.sidebar.number_input("Model Year", min_value=1990, max_value=2025, value=2018)
kms_driven = st.sidebar.number_input("KMs Driven", min_value=0, max_value=200000, value=20000)
mileage = st.sidebar.number_input("Mileage (km/l)", min_value=10, max_value=100, value=45)
cc = st.sidebar.number_input("Engine CC", min_value=50, max_value=1500, value=150)

brand = st.sidebar.selectbox("Brand", df_cleaned['brand'].unique())
owner = st.sidebar.selectbox("Owner Type", df_cleaned['owner'].unique())
location = st.sidebar.selectbox("Location", df_cleaned['location'].unique())

# --- Predict Button ---
if st.sidebar.button("Predict Price"):
    # Make user input row
    user_input = pd.DataFrame([{
        "model_year": model_year,
        "kms_driven": kms_driven,
        "mileage": mileage,
        "cc": cc,
        "brand": brand,
        "owner": owner,
        "location": location
    }])

    # Encode same as training
    user_input_encoded = pd.get_dummies(user_input, columns=['brand','owner','location'], drop_first=True)
    user_input_encoded = user_input_encoded.reindex(columns=X.columns, fill_value=0)

    # Add constant for OLS
    user_input_const = sm.add_constant(user_input_encoded, has_constant='add')

    # Prediction (back-transform log → price)
    log_pred = ols_model.predict(user_input_const)[0]
    # predicted_price = np.expm1(log_pred)
    predicted_price = log_pred

    # st.subheader(f"💰 Predicted Price: ₹{predicted_price:,.2f}")
    st.success(f"💰 Predicted Price: ₹ {predicted_price:,.2f}")

    # Show relevant bikes from dataset
    tolerance = 0.10  # ±10%
    lower, upper = predicted_price*(1-tolerance), predicted_price*(1+tolerance)

    relevant_bikes = df_cleaned[
        (df_cleaned['price'] >= lower) & 
        (df_cleaned['price'] <= upper) 
    ]
    
    
    
    relevant_bikes = relevant_bikes.sort_values('price').reset_index(drop=True)
    st.write(f"### 🔍 Found {len(relevant_bikes)} similar bikes in dataset:")
    st.dataframe(relevant_bikes[['brand', 'model_name', 'cc', 'model_year', 'owner', 'price']])
    st.info(f"Model Training RMSE: {rmse:,.2f}")
