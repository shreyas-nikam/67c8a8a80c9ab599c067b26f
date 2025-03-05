import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb

st.set_page_config(page_title="QuCreate Streamlit Lab: PDP vs ALE", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.sidebar.title("Model Interpretability Tool")
st.sidebar.markdown("Compare PDP and ALE plots to understand feature effects.")
st.title("QuLab: PDP and ALE Comparison Tool")
st.divider()

# Explanation Section
st.markdown("""
This application is designed as an educational tool to visualize and compare Partial Dependence Plots (PDPs) and Accumulated Local Effects (ALEs).
It uses a synthetic dataset to demonstrate how these interpretability methods reveal feature effects, particularly in datasets with correlated features.

**Key Features:**

*   **Model Selection:** Choose between LightGBM and XGBoost models to train on the synthetic data.
*   **Feature Analysis:** Select a feature to analyze its Partial Dependence and Accumulated Local Effects.
*   **Correlation Control:** Interact with a slider to adjust the correlation between features in the synthetic dataset and observe how it impacts PDP and ALE plots.
*   **Side-by-Side Comparison:** Dynamically view PDP and ALE plots side-by-side for immediate visual comparison.

**Learn about:**

*   **Partial Dependence Plots (PDPs):** Understand how PDPs show the average effect of a feature on the model's prediction, marginalizing over the other features.
*   **Accumulated Local Effects (ALEs):** Explore how ALE plots offer an alternative to PDPs, especially useful when features are correlated, by calculating and accumulating local effects.
*   **Feature Correlation Impact:** Visualize how varying feature correlation changes the insights provided by PDP and ALE plots.

Use the sidebar to configure the dataset, select a model, and choose a feature to analyze. Observe the interactive plots and read the explanations to deepen your understanding of model interpretability techniques.
""")

st.divider()

# --- Dataset Generation ---
st.header("1. Synthetic Dataset Generation")
st.write("Generate a synthetic dataset to explore PDP and ALE.")

n_samples = st.slider("Number of samples", min_value=100, max_value=1000, value=500, step=100, help="Adjust the size of the synthetic dataset.")
correlation_strength = st.slider("Feature Correlation Strength", min_value=0.0, max_value=1.0, value=0.5, step=0.1, help="Control the correlation between features. Higher values mean stronger correlation.")

@st.cache_data
def generate_synthetic_data(n_samples, correlation):
    """Generates a synthetic dataset with controlled correlation."""
    rng = np.random.RandomState(42)
    X = rng.rand(n_samples, 2)  # Two base features
    if correlation > 0:
        X[:, 1] = X[:, 0] * correlation + X[:, 1] * (1 - correlation) # Introduce correlation
    y = (X[:, 0] + X[:, 1]**2 + np.sin(X[:, 0] * 5)) # Non-linear relationship
    feature_names = ['Feature 1 (Linear)', 'Feature 2 (Non-linear)']
    return pd.DataFrame(X, columns=feature_names), y, feature_names

X_df, y_series, feature_names = generate_synthetic_data(n_samples, correlation_strength)
X = X_df.values
y = y_series.values

st.write("Synthetic Dataset Preview:")
st.dataframe(X_df.head())

st.markdown(f"""
**Dataset Description:**

*   **Number of Samples:** {n_samples}
*   **Features:** 2 numeric features - '{feature_names[0]}' and '{feature_names[1]}'
*   **Target Variable:** Continuous numeric target based on a non-linear function of the features.
*   **Correlation:** Feature '{feature_names[1]}' is correlated with '{feature_names[0]}' based on the selected strength.

**Formula for Target Variable (for educational purpose):**
```
y = Feature 1 + (Feature 2)^2 + sin(Feature 1 * 5)
```

**Understanding Feature Correlation:**
Feature correlation means that the features are not independent. In this synthetic dataset, we control the correlation between '{feature_names[0]}' and '{feature_names[1]}'.
When correlation is high, changes in one feature are likely to be associated with changes in the other. This can affect the interpretability of models and how PDP and ALE plots are visualized.
""")

st.divider()

# --- Model Training ---
st.header("2. Model Selection and Training")
st.write("Choose a model to train and interpret.")

model_type = st.sidebar.selectbox("Select Model Type", ["LightGBM", "XGBoost"], help="Choose between LightGBM and XGBoost gradient boosting models.")

st.sidebar.divider()

@st.cache_data
def train_model(model_type, X, y):
    """Trains a model based on selected type."""
    st.info(f"Training {model_type} model...")
    if model_type == "LightGBM":
        model = lgb.LGBMRegressor(random_state=42)
    elif model_type == "XGBoost":
        model = xgb.XGBRegressor(random_state=42)
    else:
        st.error("Invalid model type selected.")
        return None

    model.fit(X, y)
    st.success(f"{model_type} model training complete.")
    return model

model = train_model(model_type, X, y)

st.markdown(f"""
**Model Selection:**

You have selected the **{model_type}** model. This is a gradient boosting algorithm known for its performance and efficiency.

**Model Training Process:**

The selected model is trained on the synthetic dataset generated in the previous step.
The model learns the relationship between the features ('{feature_names[0]}' and '{feature_names[1]}') and the target variable 'y'.

**Gradient Boosting:**
Both LightGBM and XGBoost are gradient boosting algorithms. Gradient boosting builds models in a stage-wise fashion, like other boosting methods, but it generalizes them by allowing optimization of an arbitrary differentiable loss function.

**Why Gradient Boosting?**
Gradient boosting models are powerful and flexible, often achieving high predictive accuracy. They are also known for their ability to capture complex non-linear relationships in data, making them suitable for this demonstration.
""")

st.divider()

# --- PDP and ALE Calculation and Comparison ---
st.header("3. PDP and ALE Comparison")
st.write("Compare Partial Dependence Plots (PDP) and Accumulated Local Effects (ALE) for feature interpretability.")

selected_feature = st.selectbox("Select Feature to Analyze", feature_names, help="Choose a feature to visualize PDP and ALE plots for.")
feature_index = feature_names.index(selected_feature)

st.markdown(f"""
**Feature Selection for Analysis:**

You have selected **'{selected_feature}'** for interpretability analysis.

**Partial Dependence Plot (PDP):**

PDPs visualize the average effect of a feature on the predicted outcome of a machine learning model.

**Formula (Simplified):**

For a feature j and a value x_j, the PDP is calculated as:

```
PDP(x_j) = Average prediction over all possible values of other features, while holding feature j at x_j.
```

**Interpretation:**
The PDP curve shows how the model's prediction changes as '{selected_feature}' varies, while averaging out the effects of all other features.
It helps understand the global effect of '{selected_feature}' on the prediction.

**Accumulated Local Effects (ALE) Plot:**

ALE plots are an alternative to PDPs, designed to address issues with feature correlation. ALE plots also show feature effects but use a different approach to handle correlated features.

**Formula (Simplified):**

ALE calculates the *local effect* of changing a feature within a small interval and then *accumulates* these local effects across the feature's range.

**Interpretation:**
The ALE curve shows how the model's prediction changes as '{selected_feature}' varies, focusing on the *change* in prediction caused *specifically* by that feature, minimizing the influence of correlated features.
ALE plots are particularly useful when dealing with correlated features as they attempt to isolate the true effect of the feature of interest.

**Side-by-Side Comparison:**
By visualizing PDP and ALE plots side-by-side, you can observe how these methods differ in showing feature effects, especially in the context of feature correlation.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Partial Dependence Plot (PDP)")
    st.caption("Shows the average effect of the feature on the prediction.")
    if model:
        st.info("Generating PDP...")
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            display = PartialDependenceDisplay.from_estimator(
                model,
                X,
                [feature_index],
                feature_names=feature_names,
                ax=ax,
                centered=False,
                grid_resolution=50
            )
            ax.set_title(f"PDP for {selected_feature} ({model_type})")
            st.pyplot(fig)
            st.success("PDP generated successfully.")
        except Exception as e:
            st.error(f"Error generating PDP: {e}")
    else:
        st.warning("Model not trained yet. Please train the model first.")

with col2:
    st.subheader("Accumulated Local Effects (ALE) Plot")
    st.caption("Shows the accumulated local effects of the feature on the prediction, addressing correlation bias.")
    if model:
        st.info("Generating ALE plot...")
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            display = PartialDependenceDisplay.from_estimator(
                model,
                X,
                [feature_index],
                feature_names=feature_names,
                ax=ax,
                kind='ale',
                centered=False,
                grid_resolution=50
            )
            ax.set_title(f"ALE Plot for {selected_feature} ({model_type})")
            st.pyplot(fig)
            st.success("ALE plot generated successfully.")
        except Exception as e:
            st.error(f"Error generating ALE plot: {e}")
    else:
        st.warning("Model not trained yet. Please train the model first.")

st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption("The purpose of this demonstration is solely for educational use and illustration. "
           "To access the full legal documentation, please visit this link. Any reproduction of this demonstration "
           "requires prior written consent from QuantUniversity.")
