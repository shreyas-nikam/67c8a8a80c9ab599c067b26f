# QuLab: PDP and ALE Comparison Tool

## Description

This Streamlit application is an interactive educational tool designed to visualize and compare Partial Dependence Plots (PDPs) and Accumulated Local Effects (ALEs). It utilizes a synthetic dataset to demonstrate how these model interpretability methods reveal feature effects, especially in datasets with correlated features.

**Key Features:**

*   **Model Selection:** Choose between LightGBM and XGBoost models to train on synthetic data.
*   **Feature Analysis:** Select any feature from the dataset to analyze its Partial Dependence and Accumulated Local Effects.
*   **Correlation Control:** Adjust the correlation strength between features in the synthetic dataset using an interactive slider and observe the impact on PDP and ALE plots.
*   **Side-by-Side Plot Comparison:** Dynamically view PDP and ALE plots side-by-side for immediate visual comparison and analysis.

**Educational Objectives:**

This application helps users understand:

*   **Partial Dependence Plots (PDPs):** How PDPs illustrate the average effect of a feature on the model's prediction by marginalizing over other features.
*   **Accumulated Local Effects (ALEs):** How ALE plots offer an alternative to PDPs, particularly useful for correlated features, by focusing on local feature effects.
*   **Impact of Feature Correlation:** How varying feature correlation influences the insights provided by PDP and ALE plots.

Use the sidebar to configure the dataset parameters, select a machine learning model, and choose a feature for analysis. Observe the interactive plots and read the accompanying explanations to enhance your understanding of model interpretability techniques.

---

## Installation

To run this Streamlit application, you need to have Python installed on your system. It is recommended to use Python 3.8 or higher.

1.  **Clone the repository (or download the script):**
    If you have access to the repository, clone it using Git:
    ```bash
    git clone [repository_url]
    cd [repository_directory]
    ```
    If you only have the Python script (`streamlit_app.py` or similar), place it in a directory of your choice.

2.  **Create a virtual environment (recommended):**
    It's good practice to create a virtual environment to isolate project dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    Install the required Python packages using pip. You can create a `requirements.txt` file with the following content:
    ```
    streamlit
    pandas
    numpy
    scikit-learn
    matplotlib
    lightgbm
    xgboost
    ```
    Save this file as `requirements.txt` in the same directory as your Streamlit script and run:
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, you can install them individually:
    ```bash
    pip install streamlit pandas numpy scikit-learn matplotlib lightgbm xgboost
    ```

---

## Usage

1.  **Run the Streamlit application:**
    Navigate to the directory containing your Streamlit script (`streamlit_app.py` or the name you saved it as) in your terminal and run the following command:
    ```bash
    streamlit run streamlit_app.py
    ```
    Replace `streamlit_app.py` with the actual name of your Python script if it's different.

2.  **Access the application in your browser:**
    Streamlit will automatically open the application in your default web browser. If it doesn't, you can manually access it by navigating to the URL provided in the terminal (usually `http://localhost:8501`).

3.  **Interact with the application:**

    *   **Sidebar Controls:**
        *   **Model Selection:** In the sidebar on the left, use the "Select Model Type" dropdown to choose between "LightGBM" and "XGBoost" models.
        *   **Feature Correlation Strength:** Adjust the "Feature Correlation Strength" slider to control the correlation between the synthetic features.

    *   **Main Panel Interactions:**
        *   **Dataset Generation (Section 1):** Observe the synthetic dataset parameters (number of samples, correlation strength) and preview the generated data.
        *   **Model Selection and Training (Section 2):** Review the selected model type and understand the model training process.
        *   **PDP and ALE Comparison (Section 3):**
            *   **Feature Selection:** Use the "Select Feature to Analyze" dropdown to choose the feature for which you want to see PDP and ALE plots.
            *   **Plot Visualization:** View the Partial Dependence Plot (PDP) on the left and the Accumulated Local Effects (ALE) plot on the right. Compare the plots side-by-side and observe how they differ, especially with varying correlation strengths.
            *   Read the explanations provided for each section to understand the concepts and interpret the plots.

4.  **Experiment and Learn:**
    *   Vary the "Feature Correlation Strength" slider and observe how the PDP and ALE plots change. Notice how ALE plots are designed to be less affected by feature correlation compared to PDPs.
    *   Select different features to analyze and compare their PDP and ALE plots.
    *   Switch between LightGBM and XGBoost models and see if there are any noticeable differences in the plots for the same feature and dataset configuration.

---

## Credits

Developed and maintained by **QuantUniversity**.

[![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)](https://www.quantuniversity.com)

For more information about QuantUniversity and our educational resources, please visit [www.quantuniversity.com](https://www.quantuniversity.com).

---

## License

**Copyright © 2025 QuantUniversity. All Rights Reserved.**

This demonstration is intended solely for educational purposes and illustration. To access the full legal documentation, please visit [link to legal documentation if available, otherwise remove this part]. Any reproduction or distribution of this demonstration requires prior written consent from QuantUniversity. For licensing inquiries or permissions, please contact QuantUniversity directly.
