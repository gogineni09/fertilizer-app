# =============================
# 🌾 Smart Fertilizer Recommendation System
# =============================

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import altair as alt

# -----------------------------
# App Title
# -----------------------------
st.title("🌾 Smart Fertilizer Recommendation System")
st.write("AI-based model to predict yield and suggest the best fertilizer mix for each crop.")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("fertilizer_yield_data.csv")

st.markdown("### 📊 Dataset Preview")

# Show all 200 rows if user wants
if st.checkbox("Show full dataset"):
    st.dataframe(df)
else:
    st.dataframe(df.head(10))

st.info(f"✅ Loaded {len(df)} records successfully!")

# -----------------------------
# Encode categorical variables
# -----------------------------
le_crop = LabelEncoder()
le_region = LabelEncoder()

df["Crop"] = le_crop.fit_transform(df["Crop"])
df["Region"] = le_region.fit_transform(df["Region"])

# -----------------------------
# Prepare features and target
# -----------------------------
X = df[["Crop", "Region", "Rainfall", "pH", "N", "P", "K"]]
y = df["Yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

st.success("✅ Model trained successfully!")

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🧑‍🌾 Enter Crop Details for Prediction")

crop_name = st.selectbox("Select Crop", le_crop.classes_, key="crop_select")
region_name = st.selectbox("Select Region", le_region.classes_, key="region_select")
rain = st.number_input("Rainfall (mm)", min_value=50, max_value=2000, value=200)
ph = st.number_input("Soil pH", min_value=4.0, max_value=9.0, value=6.5)

if st.button("🔍 Predict Fertilizer Mix and Yield", key="predict_main"):
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

# -----------------------------
# 🌦️ Rainfall vs Yield Analysis
# -----------------------------
st.markdown("---")
st.subheader("🌦️ Rainfall vs Yield Analysis")
st.write("This chart shows how rainfall quantity affects crop yield across all regions and crops.")

# Scatter Chart (all crops)
scatter = alt.Chart(df).mark_circle(size=60).encode(
    x=alt.X("Rainfall", title="Rainfall (mm)"),
    y=alt.Y("Yield", title="Yield (tons/acre)"),
    color="Crop",
    tooltip=["Region", "Crop", "Rainfall", "Yield"]
).interactive()

st.altair_chart(scatter, use_container_width=True)

# Line Chart (by crop)
st.markdown("### 📈 Analyze a Specific Crop")
crop_selected = st.selectbox("Select crop to view rainfall trend:", le_crop.classes_, key="rain_chart")

# Convert crop name → encoded number for filtering
filtered = df[df["Crop"] == le_crop.transform([crop_selected])[0]]

line_chart = alt.Chart(filtered).mark_line(point=True).encode(
    x=alt.X("Rainfall", title="Rainfall (mm)"),
    y=alt.Y("Yield", title="Yield (tons/acre)"),
    color="Region",
    tooltip=["Region", "Rainfall", "Yield"]
).interactive()

st.altair_chart(line_chart, use_container_width=True)

# -----------------------------
# 🌍 Regional Rainfall Comparison
# -----------------------------
st.markdown("---")
st.subheader("🌍 Regional Rainfall vs Yield Comparison")
st.write("Compare how rainfall affects crop yield across different regions using your dataset.")

comparison_chart = alt.Chart(df).mark_line(point=True).encode(
    x=alt.X("Rainfall", title="Rainfall (mm)"),
    y=alt.Y("Yield", title="Yield (tons/acre)"),
    color=alt.Color("Region", title="Region"),
    tooltip=["Region", "Crop", "Rainfall", "Yield"]
).properties(
    width=700,
    height=400,
    title="Rainfall vs Yield for Different Regions"
).interactive()

st.altair_chart(comparison_chart, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")


st.markdown("<p style='text-align:center;'>© 2025 NIMS University | Developed by gogineni lekhana chowdary</p>", unsafe_allow_html=True)
