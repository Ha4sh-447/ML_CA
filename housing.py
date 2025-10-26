import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="California Housing Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stSlider > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ App Header ------------------
st.markdown('<p class="main-header">🏡 California Housing Price Prediction</p>', unsafe_allow_html=True)
st.markdown(
    "This interactive app demonstrates how **regularization (Ridge, Lasso, ElasticNet)** affects "
    "the prediction result using the California Housing dataset. Adjust parameters to see "
    "how the model performance changes!"
)

# ------------------ Load Dataset ------------------
@st.cache_data
def load_data():
    """Loads the California Housing dataset."""
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="MedHouseVal")
    return X, y, housing.feature_names

X, y, feature_names = load_data()

# --- Get min/max for sliders ---
lat_min = float(X['Latitude'].min())
lat_max = float(X['Latitude'].max())
lon_min = float(X['Longitude'].min())
lon_max = float(X['Longitude'].max())
pop_min = int(X['Population'].min())
pop_slider_max = min(int(X['Population'].max()), 30000) 
age_min = int(X['HouseAge'].min())
age_max = int(X['HouseAge'].max())

# ------------------ Sidebar Configuration ------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/real-estate.png", width=80)
    st.title("⚙️ Configuration Panel")
    
    st.markdown("---")
    st.header("🔧 Model Settings")
    
    model_type = st.selectbox(
        "Regularization Type",
        ("Ridge", "Lasso", "ElasticNet"),
        help="Choose the regularization method"
    )
    
    alpha = st.slider(
        "Alpha (λ) - Regularization Strength",
        min_value=0.01,
        max_value=10.0,
        value=0.1,
        step=0.01,
        help="Higher values = stronger regularization = simpler model"
    )
    
    if model_type == "ElasticNet":
        l1_ratio = st.slider(
            "L1 Ratio",
            0.0, 1.0, 0.5,
            help="0 = Ridge, 1 = Lasso, 0.5 = Equal mix"
        )
    else:
        # We pass a consistent value for caching
        l1_ratio = 0.0 if model_type == "Ridge" else 1.0 
    
    test_size = st.slider(
        "Test Set Size (%)",
        10, 50, 20,
        help="Percentage of data used for testing"
    ) / 100
    
    st.markdown("---")
    st.header("🏡 Property Features")
    
    medinc = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 3.5, step=0.1)
    house_age = st.slider("Housing Median Age", age_min, age_max, 25, step=1)
    ave_rooms = st.slider("Average Rooms", 1.0, 10.0, 5.0, step=0.1)
    ave_bedrms = st.slider("Average Bedrooms", 0.5, 2.0, 1.0, step=0.1)
    population = st.slider("Population", pop_min, pop_slider_max, 1500, step=1)
    ave_occup = st.slider("Average Occupancy", 1.0, 5.0, 3.0, step=0.1)
    latitude = st.slider("Latitude", lat_min, lat_max, 36.0, step=0.01)
    longitude = st.slider("Longitude", lon_min, lon_max, -119.0, step=0.01)


# ------------------ (OPTIMIZED) Model Training Function ------------------
@st.cache_data
def get_model_and_metrics(model_type, alpha, l1_ratio, test_size):
    """
    Splits, scales, and trains a single model.
    This function is cached, so it only re-runs when its inputs change.
    """
    # Load data (this call is cached, so it's instant)
    X_cache, y_cache, _ = load_data() 
    
    # 1. Preprocess
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_cache, y_cache, test_size=test_size, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 2. Train Model
    if model_type == "Ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "Lasso":
        model = Lasso(alpha=alpha, max_iter=10000)
    else:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)

    model.fit(X_train_scaled, y_train)
    
    # 3. Calculate Metrics
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # 4. Return everything needed
    return model, scaler, train_mse, test_mse, train_r2, test_r2

# --- Call the cached model training function ---
# This is now the *only* place the model is trained
current_model, scaler, train_mse, test_mse, train_r2, test_r2 = get_model_and_metrics(
    model_type, alpha, l1_ratio, test_size
)

# ------------------ (Cached) Function for Plot Data ------------------
@st.cache_data
def get_regression_curves(model_type, l1_ratio, test_size, data_hash):
    """
    Calculates error curves for a range of alphas.
    This function contains the CORRECT preprocessing pipeline.
    """
    # 1. Load Data
    X_cache, y_cache, feature_names_cache = load_data()
    
    # 2. Split First
    X_train_cache_raw, X_test_cache_raw, y_train_cache, y_test_cache = train_test_split(
        X_cache, y_cache, test_size=test_size, random_state=42
    )
    
    # 3. Scale Second
    scaler_cache = StandardScaler()
    X_train_scaled_cache = scaler_cache.fit_transform(X_train_cache_raw)
    X_test_scaled_cache = scaler_cache.transform(X_test_cache_raw)
    
    alphas = np.logspace(-2, 1, 50) 
    train_errors, test_errors = [], []
    
    for a in alphas:
        if model_type == "Ridge":
            model = Ridge(alpha=a)
        elif model_type == "Lasso":
            model = Lasso(alpha=a, max_iter=10000)
        else:
            model = ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=10000)
        
        model.fit(X_train_scaled_cache, y_train_cache)
        train_errors.append(mean_squared_error(y_train_cache, model.predict(X_train_scaled_cache)))
        test_errors.append(mean_squared_error(y_test_cache, model.predict(X_test_scaled_cache)))
        
    return alphas, train_errors, test_errors

# ------------------ Main Content Layout ------------------
tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "📈 Model Performance", "🔮 Prediction"])

with tab1:
    # Create a display DataFrame
    df_display = X.copy()
    df_display['MedHouseVal ($100k)'] = y
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="sub-header">Dataset Preview</p>', unsafe_allow_html=True)
        st.dataframe(df_display.head(10), use_container_width=True)
    
    with col2:
        st.markdown('<p class="sub-header">Dataset Statistics</p>', unsafe_allow_html=True)
        st.metric("Total Records", len(X))
        st.metric("Features", X.shape[1])
        st.metric("Average Value", f"${y.mean() * 100000:,.2f}")
        st.metric("Max Value", f"${y.max() * 100000:,.2f}")
    
    st.markdown('<p class="sub-header">Feature Descriptions</p>', unsafe_allow_html=True)
    st.markdown("""
    - **MedInc**: Median income in block group (in tens of thousands of dollars)
    - **HouseAge**: Median house age in block group
    - **AveRooms**: Average number of rooms per household
    - **AveBedrms**: Average number of bedrooms per household
    - **Population**: Block group population
    - **AveOccup**: Average number of household members
    - **Latitude**: Latitude of the block group
    - **Longitude**: Longitude of the block group
    - **MedHouseVal ($100k)**: Target variable - Median house value (in $100,000s)
    """)

with tab2:
    # ------------------ Display Current Metrics ------------------
    st.markdown('<p class="sub-header">Current Model Performance (at selected α)</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Train MSE", f"{train_mse:.4f}")
    with col2:
        st.metric("Test MSE", f"{test_mse:.4f}")
    with col3:
        st.metric("Train R²", f"{train_r2:.4f}")
    with col4:
        st.metric("Test R²", f"{test_r2:.4f}")
    
    # ------------------ Get Full Range for Visualization ------------------
    alphas, train_errors, test_errors = get_regression_curves(
        model_type, l1_ratio, test_size, hash(y.to_string())
    )
    
    # ------------------ Interactive Plot (MSE) ------------------
    st.markdown('<p class="sub-header">Regression MSE Curve</p>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(alphas, train_errors, label="Training Error", marker='o', linewidth=2, markersize=4, color='#2ecc71')
    ax.plot(alphas, test_errors, label="Testing Error", marker='s', linewidth=2, markersize=4, color='#e74c3c')
    
    ax.axvline(x=alpha, color='purple', linestyle='--', linewidth=2, label=f'Current α = {alpha:.2f}')
    ax.scatter([alpha], [train_mse], color='purple', s=200, zorder=5, marker='o', edgecolors='white', linewidths=2)
    ax.scatter([alpha], [test_mse], color='purple', s=200, zorder=5, marker='s', edgecolors='white', linewidths=2)
    
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (λ) - Regularization Strength", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=12, fontweight='bold')
    ax.set_title(f"{model_type} Regression — Effect of Regularization", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # ------------------ Coefficient Table ------------------
    st.markdown('<p class="sub-header">Current Model Coefficients (Weights)</p>', unsafe_allow_html=True)
    
    # Get coefficients from the single model trained outside the tabs
    coefs = current_model.coef_
    intercept = current_model.intercept_
    non_zero_features = np.sum(coefs != 0)

    st.markdown(f"""
    These are the weights the model assigns to each feature at **α = {alpha:.2f}**.
    - **{model_type}** is currently using **{non_zero_features} / {len(feature_names)}** features.
        """)

    # Create a DataFrame for the weights
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': coefs
    })
    
    # Add the intercept
    intercept_df = pd.DataFrame({'Feature': ['(Intercept)'], 'Weight': [intercept]})
    final_df = pd.concat([intercept_df, coef_df], ignore_index=True).set_index('Feature')
    
    def highlight_zeros(val):
        color = 'red' if (val == 0 and model_type != 'Ridge') else 'inherit'
        return f'color: {color}; font-weight: bold;' if color == 'red' else ''

    st.dataframe(
        final_df.style.format({'Weight': '{:,.4f}'})
                    .applymap(highlight_zeros, subset=['Weight']),
        use_container_width=True
    )

    # ------------------ Interpretation ------------------
    st.markdown('<p class="sub-header">📚 Understanding the Results</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Alpha (λ) Effect:**
        - **Low α** → Weak regularization → Complex model
            - Risk: Overfitting (high variance)
            - Training error: Low
            - Testing error: Higher (gap is large)
        
        - **High α** → Strong regularization → Simple model
            - Risk: Underfitting (high bias)
            - Training error: Higher
            - Testing error: Higher (errors converge)
        """)
    
    with col2:
        best_alpha_index = np.argmin(test_errors)
        best_alpha = alphas[best_alpha_index]
        min_test_error = test_errors[best_alpha_index]
        
        st.markdown(f"""
        **🔍 Current Model Analysis:**
        - Alpha value: **{alpha:.2f}**
        - Model type: **{model_type}**
        - Gap between train/test MSE: **{abs(train_mse - test_mse):.4f}**
        
        **💡 Optimal Alpha:**
        Based on the curve, the minimum test error (**{min_test_error:.4f}**)
        occurs at **α ≈ {best_alpha:.2f}**.
        This is the "sweet spot" that best balances bias and variance.
        """)

with tab3:
    # ------------------ Prediction ------------------
    st.markdown('<p class="sub-header">Predict House Value</p>', unsafe_allow_html=True)
    
    # Create a dictionary from sidebar inputs
    input_data_dict = {
        "MedInc": medinc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }
    
    # Create a DataFrame with the *exact* column order from training
    input_df = pd.DataFrame([input_data_dict])[feature_names]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Input Summary:**")
        st.dataframe(input_df)
    
    with col2:
        # Scale the input using the *same scaler* from the cached function
        input_scaled = scaler.transform(input_df)
        
        # Predict using the *same model* from the cached function
        predicted_cost_100k = current_model.predict(input_scaled)[0]
        predicted_cost_actual = predicted_cost_100k * 100000 
        
        st.markdown(f"""
        <div class="metric-card">
            <h2>💵 Predicted House Value</h2>
            <h1>${predicted_cost_actual:,.2f}</h1>
            <p>(Predicted as {predicted_cost_100k:.4f} in $100,000s)</p>
        </div>
        """, unsafe_allow_html=True)
        
        avg_value_100k = y.mean()
        avg_value_actual = avg_value_100k * 100000
        diff_percent = ((predicted_cost_actual - avg_value_actual) / avg_value_actual) * 100

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit | Data Science Application</p>
</div>
""", unsafe_allow_html=True)
