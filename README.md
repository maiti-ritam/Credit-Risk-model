# German Credit Risk Prediction

This project analyzes German credit data, trains a machine learning model to predict credit risk, and provides a simple web app to make live predictions.

-----

## Files

  * **`ANALYSIS_MODEL.ipynb`**:

      * Loads and explores the `german_credit_data.csv` dataset.
      * Cleans data (drops nulls) and encodes categorical features (`LabelEncoder`).
      * Trains and evaluates four models: Decision Tree, Random Forest, **Extra Trees**, and XGBoost.
      * Saves the best-performing model (`ExtraTreesClassifier`) and its encoders as `.pkl` files using `joblib`.

  * **`app.py`**:

      * A **Streamlit** web app that loads the saved `extra_trees_credit_model.pkl` and encoders.
      * Provides a simple form for a user to input applicant data (Age, Job, Housing, Credit Amount, etc.).
      * Predicts the credit risk as "GOOD" or "BAD" based on the input.

-----

## How to Run

**1. Install Dependencies**

You will need the following Python libraries. You can install them all via pip:

```bash
pip install streamlit pandas scikit-learn joblib xgboost torch matplotlib seaborn
```

**2. Generate the Model**

  * Place your `german_credit_data.csv` file in the same directory.
  * Run the `ANALYSIS_MODEL.ipynb` notebook from top to bottom. This will generate several `.pkl` files in your folder (e.g., `extra_trees_credit_model.pkl`, `Sex_encoder.pkl`, etc.).

**3. Run the App**

  * In your terminal, run the following command:
    ```bash
    streamlit run app.py
    ```
  * Your web browser will open with the interactive prediction app.
