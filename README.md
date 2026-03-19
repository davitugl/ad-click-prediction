## About Project
This project focuses on predicting digital advertisement clicks using a dataset of user behaviors. Instead of just chasing a high score, the goal was to extract actionable business insights: **Who** clicks, **when** do they click, and **why**?
The **result:** An optimized Logistic Regression model with a **99.2% AUC**, (**accuracy: 98%**) capable of cutting through the noise to identify high-conversion users.
**Smart Feature Engineering**: I replaced noisy geographical data with custom-built features like is_weekend. This single change significantly boosted the model’s predictive power.
**Model Interpretation**: I didn't stop at accuracy. Using Feature Importance analysis, I proved that user demographics (Age, Income) and timing (Weekends) are the real drivers, while the "Time of Day" was largely irrelevant.
**Optimization**: Hyperparameter tuning via GridSearchCV to ensure the model generalizes well to new, unseen data.

## Dataset from Kaggle
**Link:** (https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad/data)


## Install Dependencies
Inside the project directory, install all dependencies using Poetry:

```bash
poetry install
```

## Run the Marimo notebook:

```bash
poetry run marimo edit notebooks/ad_click_prediction.py
```

## Live Report
(https://davitugl.github.io/ad-click-prediction/)