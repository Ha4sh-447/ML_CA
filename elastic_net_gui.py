import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("ElasticNet, Ridge and Lasso comparison (Diabetes Dataset)")

alpha = st.slider("Alpha (Regularization Strength)", 0.0, 2.0, 0.1, 0.01)
l1_ratio = st.slider("L1 Ratio (ElasticNet only)", 0.0, 1.0, 0.5, 0.05)
model_choice = st.selectbox("Select Model", ["Ridge", "Lasso", "ElasticNet"])

# Load Data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Select model
if model_choice == "Ridge":
    model = Ridge(alpha=alpha)
elif model_choice == "Lasso":
    model = Lasso(alpha=alpha)
else:
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

# Train and evaluate
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**Mean Squared Error:** {mse:.3f}")
st.write(f"**R² Score:** {r2:.3f}")
st.line_chart(y_pred[:50], height=200, use_container_width=True)
