import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #  🎯 Ad Click Prediction Project
    """)
    return


@app.cell
def _():
    # Data Processing and Visualization
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    # Sklearn: Model Selection and Preprocessing
    from sklearn.model_selection import train_test_split, cross_validate
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline




    # Sklearn: Machine Learning Models
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier




    # Sklearn: Metrics and Evaluation
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, auc
    from sklearn.model_selection import GridSearchCV
    import shap

    return (
        ColumnTransformer,
        ConfusionMatrixDisplay,
        GradientBoostingClassifier,
        GridSearchCV,
        LogisticRegression,
        Pipeline,
        RandomForestClassifier,
        RocCurveDisplay,
        StandardScaler,
        classification_report,
        cross_validate,
        mo,
        np,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(pd, sns):
    sns.set_theme(style="whitegrid", palette="muted")
    advertising_df = pd.read_csv("data/raw/advertising.csv")
    return (advertising_df,)


@app.cell
def _(mo):
    # DATA DICTIONARY
    mo.md("""
    | Column | Description | Details|
    | :--- | :--- | :--- |
    | **Daily Time Spent on Site** | consumer time on site in minutes
    | **Age** | customer age in years
    | **Area Income** | Avg. Income of geographical area of consumer
    | **Daily Internet Usage** | Avg. minutes a day consumer is on the internet
    | **Ad Topic Line** | Headline of the advertisement
    | **City** | City of consumer
    | **Male** | Whether or not consumer was male
    | **Country** | Country of consumer
    | **Timestamp** | Time at which consumer clicked on Ad or closed window
    | **Clicked on Ad** | 0 or 1 indicated clicking on Ad | **(TARGET)**
    """)
    return


@app.cell
def _(advertising_df, mo):
    # LOADING & SHOWING DATA
    rows, columns = advertising_df.shape
    mo.vstack([
        mo.md("## 🎯 Ad Click Prediction Project"),
        mo.md(f"#### Total Records: **{rows}** | Total Columns: **{columns}**"),
        mo.ui.table(
            advertising_df,
            label="Advertising Data",
            selection=None,
            pagination=True
        )
    ])
    return


@app.cell
def _(advertising_df, mo, pd):
    # DATA QUALITY & PROFILING
    # Check for missing values and duplicates, unique values
    missing_values = advertising_df.isna().sum()
    duplicate_count = advertising_df.duplicated().sum()

    mo.vstack([
        mo.md("## 🔍 Data Quality"),
        mo.md(f"#### Duplicates: **{duplicate_count}**"),
        mo.md(f"#### Missing Values: **{missing_values.sum()}**"),
        mo.ui.table(
            pd.DataFrame({
                "Data Type": advertising_df.dtypes.astype(str),
                "Unique Values": advertising_df.nunique()
            }),
            selection=None
        )
    ])
    return


@app.cell
def _(advertising_df, mo, pd, plt, sns):
    # TARGET DISTRIBUTION
    target_counts = advertising_df['Clicked on Ad'].value_counts()
    target_percent = (advertising_df['Clicked on Ad'].value_counts(normalize=True) * 100).round(2).astype(str) + '%'

    target_summary = pd.DataFrame({
        "Count": target_counts,
        "Percentage (%)": target_percent
    })

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(
        data=advertising_df, 
        x='Clicked on Ad', 
        palette=['#3498db', '#e74c3c'], 
        hue='Clicked on Ad', 
        legend=False, 
        ax=ax
    )

    ax.set_title("Visual Balance (0 vs 1)")
    ax.set_xlabel("Clicked on Ad (0 = No, 1 = Yes)")
    ax.set_ylabel("Count")


    mo.vstack([
        mo.md("## 🎯 Target Variable Distribution"),
        mo.hstack([
            mo.ui.table(target_summary, selection=None),
            fig
        ], justify="start", gap=4)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🎯 Target Variable Conclusion & Strategy

    * **Perfectly Balanced Dataset:** The target variable ("Clicked on Ad") is perfectly balanced (50% vs 50%, with exactly 500 instances per class). This is the ideal scenario for model training.
    * **Evaluation Metrics:** Because the classes are equal, **Accuracy** will be a reliable primary metric for evaluating overall performance. However, to align with business objectives (e.g., optimizing ad spend), we will also analyze **Precision** and **Recall** to understand the specific types of prediction errors.
    """)
    return


@app.cell
def _(advertising_df, mo):
    # STATISTICAL SUMMARY
    stats = advertising_df.describe().T.round(2)
    mo.vstack([
        mo.md("## 📊 Statistical Overview"),
        mo.ui.table(stats, selection=None)
    ])
    return


@app.cell
def _(advertising_df, mo, plt, sns):
    # Correlation Matrix
    corr_matrix = advertising_df.corr(numeric_only=True)

    _fig, _ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=_ax
    )

    _ax.set_title("Correlation Matrix of Ad Clicks Features", pad=20, fontsize=14, weight='bold')

    mo.vstack([
        mo.md("## 🌡️ Feature Correlation Analysis"),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## 🌡️ Feature Correlation Analysis"),
        mo.md("""
        * **Primary Drivers:** `Daily Internet Usage` (-0.79) and `Daily Time Spent on Site` (-0.75) show the strongest negative correlation with ad clicks. This suggests that users who spend less time online are more likely to interact with these ads.
        * **Demographic Impact:** `Age` (0.49) has a notable positive correlation, meaning older users in this dataset tend to click on ads more frequently.
        * **Irrelevant Features:** The `Male` feature (-0.04) shows almost zero correlation, indicating that gender does not play a significant role in predicting clicks.
        """)
    ])
    return


@app.cell
def _(advertising_df, mo, sns):
    # Daily Internet Usage vs Daily Time Spent on Site vs Target
    joint_plot = sns.jointplot(
        data=advertising_df, 
        x='Daily Time Spent on Site', 
        y='Daily Internet Usage', 
        hue='Clicked on Ad', 
        palette='coolwarm', 
        alpha=0.7
    )
    joint_fig = joint_plot.fig

    mo.vstack([
        mo.md("## 📈 Daily Internet Usage vs Daily Time Spent on Site"),
        joint_fig,
        mo.md("""
        * **Cluster Separation:** The Jointplot reveals two distinct groups. The "High Usage" group (blue), who spend significant time on both the site and the internet, almost never click on the ad.
        * **Target Audience:** The "Low Usage" group (red) is our primary target. They spend less time online, and they are the ones consistently interacting with the advertisement.
        * **Model Predictability:** The classes are nearly perfectly separable. This is a strong indicator that our classification model will likely achieve very high accuracy.
    """)
    ])
    return


@app.cell
def _(advertising_df, mo, sns):
    # Age vs Time on Site Analysis vs Target
    joint_plot_age_time = sns.jointplot(
        data=advertising_df, 
        x='Age', 
        y='Daily Time Spent on Site', 
        hue='Clicked on Ad', 
        palette='coolwarm', 
        alpha=0.7,
        kind='scatter'
    )
    joint_fig_age_time = joint_plot_age_time.fig

    mo.vstack([
        mo.md("## 📈 Age vs. Time on Site Analysis"),
        joint_fig_age_time,
        mo.md("""
        * **Demographic Patterns:** There is a distinct split in behavior. Younger users with high site engagement (top-left) rarely click on ads, while older users with lower site engagement (bottom-right) are the primary clickers.
        * **Negative Correlation:** The visual confirms the negative correlation between age and time spent on site.
        * **High Separability:** The minimal overlap between classes suggests that our classification model will perform exceptionally well using these two features.
        """)
    ])
    return


@app.cell
def _(advertising_df, mo, sns):
    # Age vs Area incole vs Target
    joint_plot_age_income = sns.jointplot(
        data=advertising_df, 
        x='Age', 
        y='Area Income', 
        hue='Clicked on Ad', 
        palette='coolwarm', 
        alpha=0.7
    )
    joint_fig_age_income = joint_plot_age_income.fig

    mo.vstack([
        mo.md("## 💰 Age vs. Area Income"),
        joint_fig_age_income,
        mo.md("""
        * **Noticeable Overlap:** Unlike previous plots, there's much more mixing here. Age and Income alone don't separate the groups perfectly.
        * **General Trend:** We still see that older users with lower area income (bottom-right) are more likely to click, but it's less obvious.
        * **Key Point:** These features are useful, but the model will definitely need the "Time Spent" metrics to reach high accuracy.
        """)
    ])
    return


@app.cell
def _(advertising_df, mo):
    mo.vstack([
        advertising_df,
        mo.md(f"""
        * **Time-based Patterns:** I suspect that users are more likely to click on ads during weekends or in the evening. To test this, I am creating new features from the `Timestamp` data.
        * **Dropping Noisy Data:** Columns like **City**, **Country**, and **Ad Topic** have too many unique values. Removing them helps the model focus on important patterns and prevents it from simply "memorizing" specific records.
        """)
    ])
    return


@app.cell
def _(advertising_df, mo, pd):
    # feature Engineering: add and drop columns

    # Copy the original dataset
    processed_advertising = advertising_df.copy()

    # Time Processing (Binning)
    processed_advertising['Timestamp'] = pd.to_datetime(processed_advertising['Timestamp'])

    # Create is_weekend: 1 if it's the weekend (index 5 and 6), 0 otherwise
    processed_advertising['is_weekend'] = (processed_advertising['Timestamp'].dt.dayofweek >= 5).astype(int)

    # Create is_evening: 1 if it's evening/night (e.g., 18:00 to 23:59), 0 otherwise
    processed_advertising['is_evening'] = (processed_advertising['Timestamp'].dt.hour >= 18).astype(int)

    # drop columns
    columns_to_drop = [
        'City', 
        'Country', 
        'Timestamp', 
        'Ad Topic Line'
    ]

    processed_advertising = processed_advertising.drop(columns=columns_to_drop, errors='ignore')

    mo.ui.table(processed_advertising, label="Processed Data Preview")
    return (processed_advertising,)


@app.cell
def _(
    ColumnTransformer,
    StandardScaler,
    processed_advertising,
    train_test_split,
):
    # Train/Test Split, Scaling

    # Separating features and target
    X = processed_advertising.drop('Clicked on Ad', axis=1)
    y = processed_advertising['Clicked on Ad']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    numeric_features = [
        'Daily Time Spent on Site', 
        'Age', 
        'Area Income', 
        'Daily Internet Usage'
    ]

    binary_features = ['Male', 'is_weekend', 'is_evening']

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('bin', 'passthrough', binary_features)
        ],
        verbose_feature_names_out=False
    )

    # Output as Pandas DataFrame
    preprocessor.set_output(transform="pandas")
    return X_test, X_train, preprocessor, y_test, y_train


@app.cell
def _(
    GradientBoostingClassifier,
    LogisticRegression,
    Pipeline,
    RandomForestClassifier,
    X_train,
    cross_validate,
    mo,
    np,
    pd,
    preprocessor,
    y_train,
):
    # Models and Pipeline
    models = {
        'LogReg': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    scoring_metrics = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']

    model_comparison = []

    # Evaluating models, Cross validation
    for name, model in models.items():

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        cv_results = cross_validate(
            pipeline, 
            X_train, 
            y_train, 
            cv=5, 
            scoring=scoring_metrics,
            n_jobs=-1
        )

        model_comparison.append({
            'Model': name, 
            'ROC-AUC': np.mean(cv_results['test_roc_auc']),
            'Accuracy': np.mean(cv_results['test_accuracy']),
            'Precision': np.mean(cv_results['test_precision']),
            'Recall': np.mean(cv_results['test_recall']),
            'F1-Score': np.mean(cv_results['test_f1'])
        })



    results_df = pd.DataFrame(model_comparison).sort_values('ROC-AUC', ascending=False)
    mo.ui.table(results_df.round(4), label="Model Evaluation Results")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    * **Observation:** The baseline model (LogReg) shows quite good results (96% accuracy) and other metrics are also very good. Therefore, let's choose this simple model for its interpretability and efficiency.
    """)
    return


@app.cell
def _(
    GridSearchCV,
    LogisticRegression,
    Pipeline,
    X_train,
    mo,
    preprocessor,
    y_train,
):
    # Hyperparameter Tuning via GridSearchCV 

    # Re-initializing the pipeline
    log_reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=2000))
    ])

    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__solver': ['saga'],
        'classifier__l1_ratio': [0, 0.5, 1]
    }

    grid_search = GridSearchCV(
        estimator=log_reg_pipeline,
        param_grid=param_grid,
        cv=5,               
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_log_reg = grid_search.best_estimator_
    best_params = grid_search.best_params_


    mo.md(f"""
    ### Optimization Results
    * **(CV Accuracy):** `{grid_search.best_score_:.2%}`
    * **(C):** `{best_params['classifier__C']}`
    * **(L1 Ratio):** `{best_params['classifier__l1_ratio']}`
    """)
    return (grid_search,)


@app.cell
def _(
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    X_test,
    classification_report,
    grid_search,
    mo,
    plt,
    y_test,
):
    # Predictions on test data. Confussion Matrix, Roc Curve
    y_pred = grid_search.predict(X_test)
    report = classification_report(y_test, y_pred)

    fig_eval, ax_eval = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay.from_estimator(
        grid_search, X_test, y_test, cmap='Blues', ax=ax_eval[0]
    )
    RocCurveDisplay.from_estimator(
        grid_search, X_test, y_test, ax=ax_eval[1]
    )

    mo.vstack([
        mo.md("## Model Evaluation on Test Set"),
        mo.md(f"```\n{report}\n```"),
        mo.as_html(fig_eval)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    * The model performs excellently with 98% accuracy; other metrics look great as well. Let's now examine which features are important and how they impact the model.
    """)
    return


@app.cell
def _(grid_search, mo, pd, plt):
    # Features Importance
    names = grid_search.best_estimator_['preprocessor'].get_feature_names_out()
    coefs = grid_search.best_estimator_['classifier'].coef_[0]

    signed_features = pd.Series(coefs, index=names).sort_values(ascending=False).head(10)

    fig_sgn, ax_sgn = plt.subplots(figsize=(10, 6))

    signed_features.plot(kind='barh', color='steelblue', ax=ax_sgn)

    ax_sgn.axvline(0, color='black', linewidth=1)

    for bar in ax_sgn.patches:
        if bar.get_width() < 0:
            bar.set_color('indianred')

    ax_sgn.set_title("Top 10 Feature Weights")
    ax_sgn.set_xlabel("Coefficient Weight")
    plt.tight_layout()

    mo.vstack([
        mo.md("### **Feature Weights**"),
        mo.as_html(fig_sgn)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 🧠 Feature Analysis
    * Our hypothesis was fully confirmed: the new time features **is_weekend** had a significant impact on the model's prediction. **Age** is the strongest positive factor (higher age means more clicks), while **Daily Time Spent on Site** and **Daily Internet Usage** are the strongest negative factor.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
