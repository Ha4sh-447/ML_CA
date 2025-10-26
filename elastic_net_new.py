import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="💰",
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
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stSlider > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ App Header ------------------
st.markdown('<p class="main-header">💰 Medical Insurance Cost Prediction</p>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
This interactive app demonstrates how <b>regularization (Ridge, Lasso, ElasticNet)</b> affects 
the <b>bias–variance tradeoff</b> using real medical insurance data. Adjust parameters in real-time 
to see how the model performance changes!
</div>
""", unsafe_allow_html=True)

# ------------------ Load Dataset ------------------
@st.cache_data
def load_data():
    return pd.read_csv("./dataset/insurance.csv")

df = load_data()

# ------------------ Sidebar Configuration ------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/health-insurance.png", width=80)
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
        max_value=100.0,
        value=1.0,
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
        l1_ratio = None
    
    test_size = st.slider(
        "Test Set Size (%)",
        10, 50, 20,
        help="Percentage of data used for testing"
    ) / 100
    
    st.markdown("---")
    st.header("👤 Patient Information")
    
    age = st.slider("Age", 18, 65, 30)
    sex = st.selectbox("Sex", ("male", "female"))
    bmi = st.slider("BMI", 15.0, 45.0, 25.0, step=0.1)
    children = st.slider("Children", 0, 5, 1)
    smoker = st.selectbox("Smoker", ("yes", "no"))
    region = st.selectbox("Region", ("northeast", "northwest", "southeast", "southwest"))

# ------------------ Main Content Layout ------------------
tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "📈 Model Performance", "🔮 Prediction"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="sub-header">Dataset Preview</p>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.markdown('<p class="sub-header">Dataset Statistics</p>', unsafe_allow_html=True)
        st.metric("Total Records", len(df))
        st.metric("Features", len(df.columns) - 1)
        st.metric("Average Cost", f"${df['charges'].mean():,.2f}")
        st.metric("Max Cost", f"${df['charges'].max():,.2f}")
    
    st.markdown('<p class="sub-header">Feature Descriptions</p>', unsafe_allow_html=True)
    st.markdown("""
    - **Age**: Age of the insured person (18-65 years)
    - **Sex**: Gender of the insured (male/female)
    - **BMI**: Body Mass Index - measure of body fat
    - **Children**: Number of dependents covered
    - **Smoker**: Smoking status (yes/no)
    - **Region**: Geographic region in the US
    - **Charges**: Medical insurance costs (target variable)
    """)

with tab2:
    # ------------------ Preprocess Data ------------------
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop("charges", axis=1)
    y = df_encoded["charges"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ------------------ Train Model with Current Alpha ------------------
    if model_type == "Ridge":
        current_model = Ridge(alpha=alpha)
    elif model_type == "Lasso":
        current_model = Lasso(alpha=alpha, max_iter=10000)
    else:
        current_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    
    current_model.fit(X_train_scaled, y_train)
    y_train_pred = current_model.predict(X_train_scaled)
    y_test_pred = current_model.predict(X_test_scaled)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # ------------------ Display Current Metrics ------------------
    st.markdown('<p class="sub-header">Current Model Performance</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Train MSE", f"{train_mse:,.0f}")
    with col2:
        st.metric("Test MSE", f"{test_mse:,.0f}")
    with col3:
        st.metric("Train R²", f"{train_r2:.4f}")
    with col4:
        st.metric("Test R²", f"{test_r2:.4f}")
    
    # ------------------ Generate Full Range for Visualization ------------------
    alphas = np.logspace(-2, 2, 50)
    train_errors, test_errors = [], []
    
    for a in alphas:
        if model_type == "Ridge":
            model = Ridge(alpha=a)
        elif model_type == "Lasso":
            model = Lasso(alpha=a, max_iter=10000)
        else:
            model = ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=10000)
        
        model.fit(X_train_scaled, y_train)
        train_errors.append(mean_squared_error(y_train, model.predict(X_train_scaled)))
        test_errors.append(mean_squared_error(y_test, model.predict(X_test_scaled)))
    
    # ------------------ Interactive Plot ------------------
    st.markdown('<p class="sub-header">Bias-Variance Tradeoff Curve</p>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(alphas, train_errors, label="Training Error", marker='o', linewidth=2, markersize=4, color='#2ecc71')
    ax.plot(alphas, test_errors, label="Testing Error", marker='s', linewidth=2, markersize=4, color='#e74c3c')
    
    # Highlight current alpha
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
    
    # ------------------ Interpretation ------------------
    st.markdown('<p class="sub-header">📚 Understanding the Results</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Alpha (λ) Effect:**
        - **Low α** → Weak regularization → Complex model
          - Risk: Overfitting (high variance)
          - Training error: Low
          - Testing error: Higher
        
        - **High α** → Strong regularization → Simple model
          - Risk: Underfitting (high bias)
          - Training error: Higher
          - Testing error: Higher
        """)
    
    with col2:
        st.markdown(f"""
        **🔍 Current Model Analysis:**
        - Alpha value: **{alpha:.2f}**
        - Model type: **{model_type}**
        - Gap between train/test MSE: **{abs(train_mse - test_mse):,.0f}**
        
        **Recommendation:**
        {'✅ Good balance! Errors are close.' if abs(train_mse - test_mse) < 10000000 else '⚠️ Consider adjusting alpha to reduce the gap.'}
        """)

with tab3:
    # ------------------ Prediction ------------------
    st.markdown('<p class="sub-header">Predict Insurance Charges</p>', unsafe_allow_html=True)
    
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Input Summary:**")
        st.write(input_data)
    
    with col2:
        # Prepare input for prediction
        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
        input_scaled = scaler.transform(input_encoded)
        
        predicted_cost = current_model.predict(input_scaled)[0]
        
        st.markdown(f"""
        <div class="metric-card">
            <h2>💵 Predicted Insurance Cost</h2>
            <h1>${predicted_cost:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        avg_cost = df['charges'].mean()
        diff_percent = ((predicted_cost - avg_cost) / avg_cost) * 100
        
        st.markdown(f"""
        <br>
        **Comparison with Average:**
        - Dataset Average: ${avg_cost:,.2f}
        - Your Prediction: {'**' + str(f'{diff_percent:+.1f}%') + '**'} {'above' if diff_percent > 0 else 'below'} average
        """, unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit | Data Science Application | Bias-Variance Tradeoff Demonstration</p>
</div>
""", unsafe_allow_html=True)
