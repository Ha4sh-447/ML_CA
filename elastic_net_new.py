# -------------------------------------------------------------
# Streamlit GUI: ElasticNet, Ridge, and Lasso Interactive Demo
# Dataset: Diabetes (sklearn)
# -------------------------------------------------------------
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Regularization Regression Demo", layout="wide")

# -------------------------------------------
# Sidebar Controls
# -------------------------------------------
st.sidebar.title("⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model", ["Ridge (L2)", "Lasso (L1)", "ElasticNet (L1 + L2)"]
)
alpha = st.sidebar.slider(
    "Alpha (Regularization Strength)", 0.001, 1.0, 0.1, 0.001, format="%.3f"
)

# Conditionally display the L1 ratio slider only for ElasticNet
if "ElasticNet" in model_choice:
    l1_ratio = st.sidebar.slider("L1 Ratio (ElasticNet only)", 0.0, 1.0, 0.5, 0.05)
else:
    l1_ratio = 0  # Set a default value, though it won't be used

st.sidebar.markdown("---")
st.sidebar.title("Display Options")
show_alpha_curve = st.sidebar.checkbox("Show Alpha Effect Graphs", True)
# Conditionally show the L1 curve checkbox
if "ElasticNet" in model_choice:
    show_l1_curve = st.sidebar.checkbox("Show L1 Ratio Curve", True)
else:
    show_l1_curve = False

# -------------------------------------------
# Load, Process, and Prepare Data
# -------------------------------------------
@st.cache_data
def load_and_prep_data():
    """Loads, splits, and scales the diabetes dataset."""
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return (
        X, y, X_train_scaled, X_test_scaled, y_train, y_test,
        feature_names, scaler,
    )

(
    X_orig, y_orig, X_train, X_test, y_train, y_test,
    feature_names, scaler,
) = load_and_prep_data()

# -------------------------------------------
# Model Selection and Training
# -------------------------------------------
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

# -------------------------------------------
# Main Page Layout
# -------------------------------------------
st.title("Regularized Regression: Ridge, Lasso & ElasticNet")
st.write(
    "An interactive demo to explore how regularization techniques combat overfitting and perform feature selection."
)

st.markdown("### 🎯 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Selected Model", model_choice)
col2.metric("Mean Squared Error (MSE)", f"{mse:.3f}")
col3.metric("R² Score", f"{r2:.3f}")

# Create tabs for different views
tab1, tab2 = st.tabs(["📊 Visualization & Analysis", "🔬 Prediction Explorer"])

# ==============================================================================
# TAB 1: VISUALIZATION & ANALYSIS
# ==============================================================================
with tab1:
    st.header("Model Coefficients")
    st.write(
        f"Shows the weight assigned to each feature by the **{model_choice}** model. Lasso and ElasticNet can drive coefficients to zero, effectively performing feature selection."
    )

    # --- Improved Coefficient Plot (Horizontal Bar Chart) ---
    fig_coef, ax_coef = plt.subplots(figsize=(10, 6))
    sorted_indices = np.argsort(np.abs(model.coef_))
    sorted_coefs = model.coef_[sorted_indices]
    sorted_features = np.array(feature_names)[sorted_indices]
    
    colors = ['#d7191c' if c < 0 else '#2c7bb6' for c in sorted_coefs]
    ax_coef.barh(sorted_features, sorted_coefs, color=colors)
    ax_coef.set_title(f"{model_choice} Feature Coefficients (α={alpha})", fontsize=16)
    ax_coef.set_xlabel("Coefficient Value", fontsize=12)
    ax_coef.set_ylabel("Features", fontsize=12)
    ax_coef.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig_coef)

    st.markdown("---")

    # --- Alpha Sweep Graphs ---
    if show_alpha_curve:
        st.header("📈 Effect of Alpha on Performance and Coefficients")
        st.write(
            "These plots show how the model's performance (R²) and feature coefficients change as the regularization strength `alpha` increases."
        )

        # Pre-calculate sweep data
        alphas = np.logspace(-4, 1, 100)
        r2_scores, coefs = [], []

        for a in alphas:
            if model_choice.startswith("Ridge"):
                m = Ridge(alpha=a)
            elif model_choice.startswith("Lasso"):
                m = Lasso(alpha=a, max_iter=5000, tol=1e-2)
            else:
                m = ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=5000, tol=1e-2)
            m.fit(X_train, y_train)
            r2_scores.append(m.score(X_test, y_test))
            coefs.append(m.coef_)
        coefs = np.array(coefs)

        col_alpha1, col_alpha2 = st.columns(2)

        with col_alpha1:
            # Performance vs Alpha
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.semilogx(alphas, r2_scores, color="#c92a2a", linewidth=2)
            ax1.set_xlabel("Alpha (log scale)", fontsize=12)
            ax1.set_ylabel("R² Score", fontsize=12)
            ax1.set_title("Performance vs. Alpha", fontsize=16)
            ax1.grid(True, linestyle="--", alpha=0.6)
            ax1.axvline(x=alpha, color='black', linestyle='--', label=f'Current Alpha={alpha}')
            ax1.legend()
            st.pyplot(fig1)

        with col_alpha2:
            # Coefficient Paths
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.semilogx(alphas, coefs)
            ax2.set_xlabel("Alpha (log scale)", fontsize=12)
            ax2.set_ylabel("Coefficient Value", fontsize=12)
            ax2.set_title("Coefficient Shrinkage Path", fontsize=16)
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.axvline(x=alpha, color='black', linestyle='--', label=f'Current Alpha={alpha}')
            st.pyplot(fig2)

    # --- L1 Ratio Sweep Graph ---
    if show_l1_curve:
        st.markdown("---")
        st.header("⚖️ Effect of L1 Ratio on ElasticNet")
        st.write(
            "This plot shows how the R² score changes as you shift the balance between L1 (Lasso) and L2 (Ridge) penalties. An `l1_ratio` of 1 is pure Lasso, and 0 is pure Ridge."
        )

        l1_ratios = np.linspace(0.01, 1.0, 50)
        r2_vals = []
        for r in l1_ratios:
            m = ElasticNet(alpha=alpha, l1_ratio=r, max_iter=5000)
            m.fit(X_train, y_train)
            r2_vals.append(m.score(X_test, y_test))

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.plot(l1_ratios, r2_vals, color="#5f3dc4", linewidth=2)
        ax3.set_xlabel("L1 Ratio (0=L2 → 1=L1)", fontsize=12)
        ax3.set_ylabel("R² Score", fontsize=12)
        ax3.set_title("ElasticNet Performance vs. L1 Ratio", fontsize=16)
        ax3.axvline(x=l1_ratio, color='black', linestyle='--', label=f'Current L1 Ratio={l1_ratio}')
        ax3.legend()
        ax3.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig3)

# ==============================================================================
# TAB 2: PREDICTION EXPLORER
# ==============================================================================
with tab2:
    st.header("Make a Prediction")
    st.write(
        "Adjust the feature values below to see how they affect the model's prediction for diabetes progression one year after baseline."
    )

    # Get min, max, and mean from the original (unscaled) data for realistic slider bounds
    min_vals = X_orig.min(axis=0)
    max_vals = X_orig.max(axis=0)
    mean_vals = X_orig.mean(axis=0)

    input_data = {}
    with st.expander("Adjust Patient Feature Values", expanded=True):
        # Create two columns for the sliders
        pred_col1, pred_col2 = st.columns(2)
        
        # Loop to create sliders for each feature, alternating between columns
        for i, feature in enumerate(feature_names):
            col = pred_col1 if i < 5 else pred_col2
            input_data[feature] = col.slider(
                label=f"{feature.upper()}",
                min_value=float(min_vals[i]),
                max_value=float(max_vals[i]),
                value=float(mean_vals[i]),
                step=0.001 if "s" in feature else 1.0, # Finer steps for serum measurements
                format="%.3f"
            )

    # --- Prediction Logic ---
    user_inputs = np.array([input_data[f] for f in feature_names]).reshape(1, -1)
    
    # Scale the user inputs using the SAME scaler fitted on the training data
    scaled_inputs = scaler.transform(user_inputs)
    
    # Make prediction
    prediction = model.predict(scaled_inputs)[0]

    st.markdown("### Prediction Result")
    st.info(
        f"The predicted quantitative measure of disease progression is:",
        icon="🩺"
    )
    st.metric(label="Predicted Score", value=f"{prediction:.2f}")

# ==============================================================================
# FINAL SECTION: THEORY
# ==============================================================================
st.markdown("---")
st.header("🧠 Conceptual Insights")
st.write(
    """
-   **Ridge (L2 Regularization)**: Shrinks all coefficient values towards zero, but never makes them exactly zero. It's great for reducing model complexity and handling multicollinearity (when features are correlated).
-   **Lasso (L1 Regularization)**: Can shrink some coefficients *exactly* to zero. This makes it a powerful tool for automatic **feature selection**, as it effectively removes irrelevant features from the model.
-   **ElasticNet (L1 + L2)**: A combination of both. The `l1_ratio` parameter controls the blend. It gets the feature selection benefit from Lasso and the stability of Ridge, which is especially useful when you have many correlated features.
-   **Alpha (α)**: The overall regularization strength. A higher `alpha` means a stronger penalty, leading to smaller coefficients, higher bias, and lower variance (simpler model).
"""
)
