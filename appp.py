import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.title("🌾 Smart Fertilizer Recommendation System")
st.write("AI-based model to predict yield and suggest the best fertilizer mix for each crop.")

df = pd.read_csv("fertilizer_yield_data.csv")
st.subheader("📊 Sample of Your Data")
st.dataframe(df.head())

le_crop = LabelEncoder()
le_region = LabelEncoder()
df["Crop"] = le_crop.fit_transform(df["Crop"])
df["Region"] = le_region.fit_transform(df["Region"])

X = df[["Crop", "Region", "Rainfall", "pH", "N", "P", "K"]]
y = df["Yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

st.success("✅ Model trained successfully!")

st.subheader("🧑‍🌾 Enter Crop Details for Prediction")
crop_name = st.selectbox("Select Crop", le_crop.classes_)
region_name = st.selectbox("Select Region", le_region.classes_)
rain = st.number_input("Rainfall (mm)", min_value=50, max_value=500, value=200)
ph = st.number_input("Soil pH", min_value=4.0, max_value=9.0, value=6.5)

if st.button("🔍 Predict Fertilizer Mix and Yield"):
    crop_val = le_crop.transform([crop_name])[0]
    region_val = le_region.transform([region_name])[0]
    subset = df[(df["Crop"] == crop_val) & (df["Region"] == region_val)]
    if len(subset) > 0:
        N_mean = round(subset["N"].mean(), 2)
        P_mean = round(subset["P"].mean(), 2)
        K_mean = round(subset["K"].mean(), 2)
    else:
        N_mean, P_mean, K_mean = 80, 40, 40
    pred = model.predict([[crop_val, region_val, rain, ph, N_mean, P_mean, K_mean]])
    st.success(f"🌿 Suggested NPK Ratio → N={N_mean}, P={P_mean}, K={K_mean}")
    st.info(f"📈 Predicted Yield → {round(pred[0], 2)} tons/acre")
