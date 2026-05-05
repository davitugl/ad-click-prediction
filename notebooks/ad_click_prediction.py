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
        shap,
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
    # Check duplicates and missing values
    def check_missing_duplicates(df: pd.DataFrame) -> dict:
        return {
            "duplicates": df.duplicated().sum(),
            "total_missing": df.isna().sum().sum()
        }

    miss_dupl_stats = check_missing_duplicates(advertising_df)

    mo.vstack([
        mo.md(f"**Duplicates:** {miss_dupl_stats['duplicates']}"),
        mo.md(f"**Missing Values:** {miss_dupl_stats['total_missing']}")
    ])
    return


@app.cell
def _(advertising_df, mo, pd):
    # Column types and unique values
    def check_schema(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "Dtype": df.dtypes.astype(str),
            "Unique Values": df.nunique()
        })
    # Get schema stats
    schema_info = check_schema(advertising_df)

    mo.vstack([
        mo.md("### 📑 Column Analysis"),
        mo.ui.table(schema_info)
    ])
    return


@app.cell
def _(advertising_df, mo, pd, plt, sns):
    # Target distribution
    def analyze_target(df: pd.DataFrame, column: str):
        counts = df[column].value_counts()
        percent = (df[column].value_counts(normalize=True) * 100).round(2)
        summary = pd.DataFrame({"Count": counts, "Percentage (%)": percent})

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x=column, ax=ax, palette="viridis", hue=column, legend=False)
        ax.set_title(f"Distribution: {column}")
        plt.close()

        return summary, fig

    target_summary, target_plot = analyze_target(advertising_df, 'Clicked on Ad')

    mo.vstack([
        mo.md(f"## 🎯 Target Variable Distribution"),
        mo.hstack([
            mo.ui.table(target_summary, selection=None),
            target_plot
        ], justify="start", gap=2)
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
    # Statistical summary
    stats_overview = advertising_df.describe().T.round(2)
    mo.vstack([
        mo.md("## 📊 Statistical Overview"),
        mo.ui.table(stats_overview, selection=None)
    ])
    return


@app.cell
def _(advertising_df, mo, pd, plt, sns):
    # Correlation Matrix
    def analyze_correlations(df: pd.DataFrame):
        corr_matrix = df.corr(numeric_only=True)
    
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            linewidths=0.5, 
            ax=ax
        )
        ax.set_title("Feature Correlation Matrix", pad=20, weight='bold')
        plt.close()

        return corr_matrix, fig

    # results
    correlations, corr_plot = analyze_correlations(advertising_df)

    # Display the interface
    mo.vstack([
        mo.md("## 🌡️ Feature Correlation Analysis"),
        corr_plot
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
def _(advertising_df, mo, plt, sns):
    # Daily Internet Usage vs Daily Time Spent on Site vs Target
    def get_usage_jointplot(df, x, y, hue):
        # Creating a jointplot to separate classes
        joint_plot = sns.jointplot(
            data=df, 
            x=x, 
            y=y, 
            hue=hue, 
            palette='coolwarm', 
            alpha=0.7
        )
    
        plt.close() 
        return joint_plot.fig

    # Prep. visual
    usage_fig = get_usage_jointplot(
        advertising_df, 
        'Daily Time Spent on Site', 
        'Daily Internet Usage', 
        'Clicked on Ad'
    )

    mo.vstack([
        mo.md("## 📈 Daily Internet Usage vs Daily Time Spent on Site"),
        usage_fig,
        mo.md("""
        * **Cluster Separation:** The Jointplot reveals two distinct groups. The "High Usage" group (blue), who spend significant time on both the site and the internet, almost never click on the ad.

        * **Target Audience:** The "Low Usage" group (red) is our primary target. They spend less time online, and they are the ones consistently interacting with the advertisement.

        * **Model Predictability:** The classes are nearly perfectly separable. This is a strong indicator that our classification model will likely achieve very high accuracy.

    """)
    ])
    return


@app.cell
def _(advertising_df, mo, plt, sns):
    # Age vs Time on Site Analysis vs Target
    def get_age_usage_plot(df, x, y, hue):
        plot = sns.jointplot(
            data=df, 
            x=x, 
            y=y, 
            hue=hue, 
            palette='coolwarm', 
            alpha=0.7, 
            kind='scatter'
        )
    
        plt.close()
        return plot.fig

    age_site_fig = get_age_usage_plot(
        advertising_df, 
        'Age', 
        'Daily Time Spent on Site', 
        'Clicked on Ad'
    )

    mo.vstack([
        mo.md("## 📈 Age vs. Time on Site"),
        age_site_fig,
        mo.md("""

        * **Demographic Patterns:** There is a distinct split in behavior. Younger users with high site engagement (top-left) rarely click on ads, while older users with lower site engagement (bottom-right) are the primary clickers.

        * **Negative Correlation:** The visual confirms the negative correlation between age and time spent on site.

        * **High Separability:** The minimal overlap between classes suggests that our classification model will perform exceptionally well using these two features.

        """)
    ])
    return


@app.cell
def _(advertising_df, mo, plt, sns):
    # Age vs Area incole vs Target

    def get_age_income_plot(df, x, y, hue):
        plot = sns.jointplot(
            data=df, 
            x=x, 
            y=y, 
            hue=hue, 
            palette='coolwarm', 
            alpha=0.7
        )
    
        plt.close()
        return plot.fig

    age_income_fig = get_age_income_plot(
        advertising_df, 
        'Age', 
        'Area Income', 
        'Clicked on Ad'
    )

    mo.vstack([
        mo.md("## 💰 Age vs. Area Income"),
        age_income_fig,
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

    def process_advertising_data(df: pd.DataFrame) -> pd.DataFrame:
        processed_df = df.copy()
    
        # Convert Timestamp to datetime
        timestamp = pd.to_datetime(processed_df['Timestamp'])
    
        # Feature Engineering: is_weekend and is_evening
        processed_df['is_weekend'] = (timestamp.dt.dayofweek >= 5).astype(int)
        processed_df['is_evening'] = (timestamp.dt.hour >= 18).astype(int)
    
        # Drop unnecessary columns
        to_drop = ['City', 'Country', 'Timestamp', 'Ad Topic Line']
        return processed_df.drop(columns=to_drop, errors='ignore')

    # Prepare the processed dataset
    processed_advertising = process_advertising_data(advertising_df)

    mo.vstack([
        mo.md("## 🛠️ Processed Data"),
        mo.ui.table(processed_advertising, label="Processed Data Preview")
    ])
    return (processed_advertising,)


@app.cell
def _(
    ColumnTransformer,
    StandardScaler,
    pd,
    processed_advertising,
    train_test_split,
):
    # Train/Test Split, Scaling, Preprocessor
    # Train/Test Split, Scaling, Preprocessor
    def prepare_ml_data(df: pd.DataFrame, target: str):
        # Data splitting
        X = df.drop(columns=[target])
        y = df[target]
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
        # Preprocessor setup
        num_cols = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']
        bin_cols = ['Male', 'is_weekend', 'is_evening']
    
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('bin', 'passthrough', bin_cols)
            ],
            verbose_feature_names_out=False
        ).set_output(transform="pandas")
    
        return X_train, X_test, y_train, y_test, preprocessor

    # Calling the function and assigning variables
    X_train, X_test, y_train, y_test, preprocessor = prepare_ml_data(processed_advertising, 'Clicked on Ad')
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
    def evaluate_models(X, y, preprocessor):
        # Define models
        models = {
            'LogReg': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
    
        # List of metrics
        metrics = ['roc_auc', 'accuracy', 'precision', 'f1']
        results = []

        for name, model in models.items():
            # Create pipeline and validate
            pipe = Pipeline([('pre', preprocessor), ('clf', model)])
            cv = cross_validate(pipe, X, y, cv=5, scoring=metrics, n_jobs=-1)
        
            # Collecting results
            results.append({
                'Model': name,
                'ROC-AUC': np.mean(cv['test_roc_auc']),
                'Accuracy': np.mean(cv['test_accuracy']),
                'Precision': np.mean(cv['test_precision']),
                'F1-Score': np.mean(cv['test_f1'])
            })
    
        return pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)

    # Evaluate models
    comparison_df = evaluate_models(X_train, y_train, preprocessor)

    mo.vstack([
        mo.md("## 🏆 Model Evaluation Results"),
        mo.ui.table(comparison_df.round(4), selection=None)
    ])
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

    def log_reg_tuning(X, y, preprocessor):
    
        # Define the pipeline
        pipe = Pipeline([
            ('pre', preprocessor),
            ('clf', LogisticRegression(random_state=42, max_iter=2000))
        ])

        # Parameter grid exactly as requested
        param_grid = {
            'clf__C': [0.1, 1, 10, 100],
            'clf__solver': ['saga'],
            'clf__l1_ratio': [0, 0.5, 1]
        }

        # Grid Search configuration
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
    
        return grid.fit(X, y)

    # Running the optimization
    tuning_results = log_reg_tuning(X_train, y_train, preprocessor)

    # Displaying the best parameters
    best_params = tuning_results.best_params_

    mo.md(f"""
    ### 🚀 Optimization Results
    * **CV Accuracy:** `{tuning_results.best_score_:.2%}`
    * **Best C:** `{best_params['clf__C']}`
    * **Best L1 Ratio:** `{best_params['clf__l1_ratio']}`
    """)
    return (tuning_results,)


@app.cell
def _(
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    X_test,
    classification_report,
    mo,
    plt,
    tuning_results,
    y_test,
):
    # Predictions on test data. Confussion Matrix, Roc Curve
    def evaluate_final_model(model, X_test, y_test):
        # Predictions and metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)

        # Visual
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues', ax=ax[0])
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax[1])
    
        plt.tight_layout()
        plt.close()
    
        return report, fig

    # Run evaluation
    final_report, eval_fig = evaluate_final_model(tuning_results.best_estimator_, X_test, y_test)

    # Display 
    mo.vstack([
        mo.md("## Final Evaluation"),
        mo.md(f"```\n{final_report}\n```"),
        eval_fig
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    * The model performs excellently with 98% accuracy; other metrics look great as well. Let's now examine which features are important and how they impact the model.
    """)
    return


@app.cell
def _(X_test, mo, plt, shap, tuning_results):
    # Features Importance with shap

    def shap_impact(best_pipeline, X_input):
    
        X_tx = best_pipeline['pre'].transform(X_input)
    
        explainer = shap.Explainer(best_pipeline['clf'], X_tx)
        shap_values = explainer(X_tx)
    
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
    
        return plt.gcf()

    shap_plot = shap_impact(tuning_results.best_estimator_, X_test)

    # შედეგის გამოტანა ორ ენაზე
    mo.vstack([
        mo.md("## 🐝 SHAP: Impact Analysis"),
        shap_plot,
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
