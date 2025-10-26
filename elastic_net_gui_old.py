import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Insurance Cost Prediction - Regularized Regression", layout="wide")

# ==============================================================
# Load and preprocess dataset
# ==============================================================
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("./dataset/insurance.csv")
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("charges", axis=1)
    y = df["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return df, X, y, X_train, X_test, y_train, y_test, scaler

df, X, y, X_train, X_test, y_train, y_test, scaler = load_and_prep_data()

# ==============================================================
# Sidebar: Model Settings
# ==============================================================
st.sidebar.title("⚙️ Model Configuration")
model_choice = st.sidebar.selectbox("Select Model", ["Ridge (L2)", "Lasso (L1)", "ElasticNet (L1 + L2)"])
alpha = st.sidebar.slider("Alpha (Regularization Strength)", 0.001, 100.0, 1.0, 0.1)

if "ElasticNet" in model_choice:
    l1_ratio = st.sidebar.slider("L1 Ratio (ElasticNet only)", 0.0, 1.0, 0.5, 0.05)
else:
    l1_ratio = 0

# ==============================================================
# Train model
# ==============================================================
if model_choice.startswith("Ridge"):
    model = Ridge(alpha=alpha)
elif model_choice.startswith("Lasso"):
    model = Lasso(alpha=alpha, max_iter=5000)
else:
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ==============================================================
# Main Layout
# ==============================================================
st.title("💰 Regularized Regression on Medical Insurance Dataset")
st.write("Explore how Ridge, Lasso, and ElasticNet affect model performance and bias–variance tradeoff on real insurance cost data.")

st.markdown("### 🎯 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Model", model_choice)
col2.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
col3.metric("R² Score", f"{r2:.3f}")

# ==============================================================
# Tabs
# ==============================================================
tab1, tab2 = st.tabs(["📊 Visualization & Bias–Variance", "🔮 Prediction Explorer"])

# ==============================================================
# TAB 1: Visualization & Bias–Variance
# ==============================================================
with tab1:
    st.header("📈 Training vs Testing Error (Bias–Variance Tradeoff)")

    alphas = np.logspace(-2, 2, 20)
    train_errors, test_errors = [], []

    for a in alphas:
        if model_choice.startswith("Ridge"):
            m = Ridge(alpha=a)
        elif model_choice.startswith("Lasso"):
            m = Lasso(alpha=a, max_iter=10000)
        else:
            m = ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=10000)

        m.fit(X_train, y_train)
        y_train_pred = m.predict(X_train)
        y_test_pred = m.predict(X_test)
        train_errors.append(mean_squared_error(y_train, y_train_pred))
        test_errors.append(mean_squared_error(y_test, y_test_pred))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(alphas, train_errors, label="Training Error", marker="o")
    ax.plot(alphas, test_errors, label="Testing Error", marker="s")
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (Regularization Strength)")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title(f"{model_choice} — Bias–Variance Tradeoff")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)

    st.markdown("""
    **Interpretation:**
    - Low α → low bias, high variance → model may overfit  
    - High α → high bias, low variance → model may underfit  
    - The sweet spot minimizes the testing error curve.
    """)

# ==============================================================
# TAB 2: Prediction Explorer
# ==============================================================
with tab2:
    st.header("🩺 Predict Insurance Charges")
    st.write("Adjust patient information to estimate their expected annual medical insurance cost.")

    # UI for user input
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 65, 30, step=1)
        bmi = st.slider("BMI (Body Mass Index)", 15.0, 45.0, 25.0, step=0.1)
        children = st.slider("Number of Children", 0, 5, 1)
    with col2:
        sex = st.selectbox("Sex", ("male", "female"))
        smoker = st.selectbox("Smoker", ("yes", "no"))
        region = st.selectbox("Region", ("northeast", "northwest", "southeast", "southwest"))

    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)[0]
    st.metric(label="Predicted Annual Insurance Cost", value=f"${prediction:,.2f}")

# ==============================================================
# Dataset Information
# ==============================================================
with st.expander("ℹ️ Dataset Information"):
    st.write("""
    **Dataset:** Medical Cost Personal Dataset (Kaggle)  
    **Rows:** 1,338  
    **Target Variable:** `charges` (annual medical insurance cost in USD)  
    **Features:**  
    - `age`: Age of the insured person  
    - `sex`: Gender (`male`/`female`)  
    - `bmi`: Body mass index (health indicator)  
    - `children`: Number of dependents  
    - `smoker`: Whether the person is a smoker (`yes`/`no`)  
    - `region`: Residential region (`northeast`, `northwest`, `southeast`, `southwest`)  

    This dataset is ideal for regression and for studying how **regularization** affects
    the **bias–variance tradeoff** in predictive modeling.
    """)

# ==============================================================
# Theory Section
# ==============================================================
st.markdown("---")
st.header("🧠 Understanding Regularization and Alpha")
st.write("""
- **Ridge (L2 Regularization):** Shrinks coefficients smoothly; helps prevent overfitting by reducing model complexity but keeps all features.
- **Lasso (L1 Regularization):** Shrinks some coefficients to exactly zero; performs **feature selection** by removing less useful predictors.
- **ElasticNet (L1 + L2):** Balances both; controlled by `l1_ratio`:
  - `l1_ratio = 0` → behaves like Ridge  
  - `l1_ratio = 1` → behaves like Lasso  
  - `0 < l1_ratio < 1` → mix of both  
- **Alpha (α):** The overall regularization strength present in all three methods:
  - In **Ridge**, α scales the L2 penalty term (sum of squares of coefficients)
  - In **Lasso**, α scales the L1 penalty term (sum of absolute values)
  - In **ElasticNet**, α scales the combined L1 + L2 penalty  
  Higher α = stronger regularization → higher bias, lower variance.
""")
