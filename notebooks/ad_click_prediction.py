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
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline




    # Sklearn: Machine Learning Models
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier




    # Sklearn: Metrics and Evaluation
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
    from sklearn.model_selection import GridSearchCV
    import shap

    return (
        ColumnTransformer,
        GradientBoostingClassifier,
        GridSearchCV,
        LogisticRegression,
        RandomForestClassifier,
        StandardScaler,
        accuracy_score,
        auc,
        classification_report,
        confusion_matrix,
        mo,
        pd,
        plt,
        roc_auc_score,
        roc_curve,
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
def _(advertising_df):
    advertising_df
    return


@app.cell
def _(advertising_df, pd):
    # feature Engineering

    # Copy the original dataset
    processed_advertising = advertising_df.copy()

    # Time Processing (Binning)
    processed_advertising['Timestamp'] = pd.to_datetime(processed_advertising['Timestamp'])

    # Create is_weekend: 1 if it's the weekend (index 5 and 6), 0 otherwise
    processed_advertising['is_weekend'] = (processed_advertising['Timestamp'].dt.dayofweek >= 5).astype(int)

    # Create is_evening: 1 if it's evening/night (e.g., 18:00 to 23:59), 0 otherwise
    processed_advertising['is_evening'] = (processed_advertising['Timestamp'].dt.hour >= 18).astype(int)

    columns_to_drop = [
        'City', 
        'Country', 
        'Timestamp', 
        'Ad Topic Line'
    ]

    processed_advertising = processed_advertising.drop(columns=columns_to_drop, errors='ignore')

    processed_advertising
    return (processed_advertising,)


@app.cell
def _(
    ColumnTransformer,
    StandardScaler,
    mo,
    processed_advertising,
    train_test_split,
):
    # Data Preprocessing Pipeline: Train/Test Split, Scaling

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

    # Fit & Transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    mo.md(f"""
    * **Training Data:** {X_train_processed.shape}
    * **Testing Data:** {X_test_processed.shape}
    """)
    return X_test_processed, X_train_processed, y_test, y_train


@app.cell
def _(
    LogisticRegression,
    X_test_processed,
    X_train_processed,
    accuracy_score,
    mo,
    roc_auc_score,
    y_test,
    y_train,
):
    # baseline Model (log regression)
    # initialize and train our simplest, baseline model
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_processed, y_train)

    # predict
    y_pred = log_reg.predict(X_test_processed)

    # probabilities for the ROC-AUC metric
    y_prob = log_reg.predict_proba(X_test_processed)[:, 1] 

    # calc ROC-AUC
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    mo.md(f"""
    ### 🚀 First Test Result: Logistic Regression

    * **Accuracy:** `{acc:.4f}`
    * **ROC-AUC Score:** `{roc_auc:.4f}` The closer to 1.0, the better

    """)
    return


@app.cell
def _(classification_report, confusion_matrix, mo, plt, roc_auc_score, sns):
    # Create a universal function to evaluate all our models!
    def evaluate_model(model, name, X_test, y_test):

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] 

        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        _fig, _ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=_ax,
                    xticklabels=['Clicked (1)', 'Not Clicked (0)'],
                    yticklabels=['Clicked (1)', 'Not Clicked (0)'])
        _ax.set_title(f'Confusion Matrix: {name}', weight='bold')
        plt.tight_layout()

        return mo.vstack([
            mo.md(f"### 🤖 Model: {name}"),
            mo.md(f"**ROC-AUC Score:** `{roc_auc:.4f}`"),
            mo.md(f"**Classification Report:**\n```text\n{report}\n```"),
            _fig,
            mo.md("---")
        ])


    return (evaluate_model,)


@app.cell
def _(
    GradientBoostingClassifier,
    LogisticRegression,
    RandomForestClassifier,
    X_test_processed,
    X_train_processed,
    evaluate_model,
    mo,
    y_test,
    y_train,
):
    # Initialization Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results_ui = []
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        ui_element = evaluate_model(model, name, X_test_processed, y_test)
        results_ui.append(ui_element)

    # display all reports together
    mo.vstack(results_ui)
    return


@app.cell
def _(mo):
    mo.md("""
    * **Observation:** The simplest model (Logistic Regression) easily beat the complex ones, hitting 97.5% accuracy. It caught almost all real clicks flawlessly and saved our ad budget.
    * **The "Why":** Our dataset is small and highly logical. In these cases, complex models overcomplicate things and get confused by noise (overfitting). A simple model just draws a straight line and hits the target.
    * **Next Step:** Try tuning the hyperparameters. Since this model is completely transparent, we can look inside its "brain". Next, we’ll find out the most important thing: what exactly makes a user click the ad — age, income, or time spent on the site?
    """)
    return


@app.cell
def _(GridSearchCV, LogisticRegression, X_train_processed, mo, y_train):
    # Hyperparameter Tuning via GridSearchCV 
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],         
        'l1_ratio': [0, 1],              
        'solver': ['liblinear'],      
        'max_iter': [1000]              
    }

    # grid_search
    grid_search = GridSearchCV(
        estimator=LogisticRegression(random_state=42),
        param_grid=param_grid,
        cv=5,               
        scoring='accuracy',
        n_jobs=-1,          
        verbose=1 
    )

    # training
    grid_search.fit(X_train_processed, y_train)

    best_log_reg = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # display
    mo.md(f"""
    * **Best Parameters:** `{best_params}`
    * **Cross-Validation Accuracy:** `{best_score:.4f}`

    Now we can use this optimized model **{best_log_reg}** for the final evaluation and to extract the weights!!
    """)
    return (best_log_reg,)


@app.cell
def _(
    X_test_processed,
    X_train_processed,
    best_log_reg,
    evaluate_model,
    mo,
    pd,
    plt,
    sns,
    y_test,
):
    # final evaluation on the test set
    evaluation_ui = evaluate_model(best_log_reg, "Optimized Logistic Regression (C=0.1)", X_test_processed, y_test)

    # Extracting model weights and feature names
    feature_names = X_train_processed.columns
    coefficients = best_log_reg.coef_[0]

    # Sorting weights in a DataFrame for visualization
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': coefficients
    }).sort_values(by='Weight', ascending=False)

    # Feature Importances
    fig_weights, ax_weights = plt.subplots(figsize=(8, 6))

    # Blue for positive impact, Red for negative
    colors = ['#1f77b4' if w > 0 else '#d62728' for w in coef_df['Weight']]
    sns.barplot(x='Weight', y='Feature', hue='Feature', data=coef_df, palette=colors, ax=ax_weights, legend=False)

    ax_weights.set_title('What drives the click?', weight='bold', fontsize=14, pad=15)
    ax_weights.set_xlabel('Feature Weight')
    ax_weights.set_ylabel('')

    ax_weights.axvline(0, color='black', linewidth=1)
    plt.tight_layout()

    mo.vstack([
        evaluation_ui,
        mo.md("""
        ### ⚖️ What Drives the Decision?
        * **Right side (Blue):** Factors that increase the likelihood of clicking.
        * **Left side (Red):** Factors that decrease the likelihood of clicking.
        """),
        fig_weights
    ])
    return


@app.cell
def _(X_test_processed, auc, best_log_reg, mo, plt, roc_curve, y_test):
    # Roc-Auc curve

    # Getting probabilities
    y_probs_final = best_log_reg.predict_proba(X_test_processed)[:, 1]

    # Computing ROC & AUC
    fpr_final, tpr_final, _thresholds_final = roc_curve(y_test, y_probs_final)
    roc_auc_value = auc(fpr_final, tpr_final)

    # Visualization
    fig_roc_final, ax_roc_final = plt.subplots(figsize=(8, 6))

    ax_roc_final.plot(fpr_final, tpr_final, color='#1f77b4', lw=3, label=f'ROC Curve (AUC = {roc_auc_value:.4f})')
    ax_roc_final.plot([0, 1], [0, 1], color='#7f7f7f', lw=2, linestyle='--')

    ax_roc_final.set_xlabel('False Positive Rate')
    ax_roc_final.set_ylabel('True Positive Rate')
    ax_roc_final.set_title('ROC Curve - Optimized Model', weight='bold')
    ax_roc_final.legend(loc="lower right")
    ax_roc_final.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.vstack([
        mo.md(f"""
        ### 🛡️ ROC-AUC
        **Score:** `{roc_auc_value:.4f}`
        """),
        fig_roc_final
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    * **Core Drivers:** Who clicks the most? Older users (`Age` has the highest positive impact). Who ignores the ad? High-income (`Area Income`) and highly active digital users (those with high `Daily Time on Site` and `Daily Internet Usage`).
    * **The Weekend Effect:** Our hypothesis was spot on! The `is_weekend` feature proves that ads get significantly more clicks on Saturdays and Sundays. A clear signal to reallocate the marketing budget.
    * **Time & Gender Impact:** Gender has a minor negative impact, while evening hours (`is_evening`) show almost zero effect.
    * **A Cleaner Model:** Removing the geographical noise (countries) resulted in a much more stable, interpretable, and production-ready model.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
