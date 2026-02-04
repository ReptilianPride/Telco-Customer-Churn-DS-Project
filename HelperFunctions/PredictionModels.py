from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split


import lightgbm as lgb
from catboost import CatBoostClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def RandomForestClassifierModel(X,y):
    # ---------------------------
    # Step 1: Encode categorical features
    # ---------------------------
    cat_features = X.select_dtypes(include=['category']).columns.tolist()

    X_encoded = X.copy()
    for col in cat_features:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])

    # ---------------------------
    # Step 2: Split data
    # ---------------------------
    xtrain, xtest, ytrain, ytest = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------
    # Step 3: Define Random Forest model
    # ---------------------------
    # class_weight='balanced' gives higher weight to minority class for high recall
    model = RandomForestClassifier(
        n_estimators=500,         # number of trees
        max_depth=10,             # similar to your LightGBM depth
        random_state=42,
        class_weight='balanced',  # prioritize minority class to improve recall
        n_jobs=-1                 # use all CPU cores
    )

    # ---------------------------
    # Step 4: Train the model
    # ---------------------------
    model.fit(xtrain, ytrain)

    # ---------------------------
    # Step 5: Make predictions
    # ---------------------------
    y_pred_prob = model.predict_proba(xtest)[:, 1]   # probability for positive class

    # Optional: adjust threshold for higher recall
    threshold = 0.4  # lower than 0.5 to increase recall
    y_pred = (y_pred_prob >= threshold).astype(int)

    # ---------------------------
    # Step 6: Evaluate
    # ---------------------------
    conf_matrix = confusion_matrix(ytest, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:\n", classification_report(ytest, y_pred))

    return model


def LightGbmModel(X,y):
    # Step 1: Encode categorical features
    cat_features = X.select_dtypes(include=['category']).columns.tolist()

    X_encoded = X.copy()
    for col in cat_features:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])


    # Step 2: Split data
    xtrain, xtest, ytrain, ytest = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )


    # Step 3: Create LightGBM datasets
    train_data = lgb.Dataset(xtrain, label=ytrain, categorical_feature=cat_features, free_raw_data=False)
    test_data = lgb.Dataset(xtest, label=ytest, reference=train_data, categorical_feature=cat_features, free_raw_data=False)


    # Step 4: Define parameters
    params = {
        'objective': 'binary',           # 'multiclass' if more than 2 classes
        'metric': 'binary_logloss',      # 'multi_logloss' for multiclass
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 1023,
        'max_depth': 10,
        'verbose': -1,
        'seed': 42,
        'device': 'gpu'
    }


    # Step 5: Train the model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
    )


    # Step 6: Make predictions
    y_pred_prob = model.predict(xtest)
    y_pred = (y_pred_prob >= 0.5).astype(int)  # For binary classification


    # Step 7: Evaluate
    conf_matrix = confusion_matrix(ytest, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:\n", classification_report(ytest, y_pred))

    return model


def CatBoostModel(X,y):

    xtrain,xtest,ytrain,ytest=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
    )

    cat_features = X.select_dtypes(include=['category']).columns.tolist()

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.01,
        depth=10,
        eval_metric='Recall',
        random_seed=42,
        task_type='GPU',
        devices='0',       
        verbose=200
    )

    model.fit(xtrain, ytrain, cat_features=cat_features, eval_set=(xtest, ytest))

    y_pred = model.predict(xtest)

    conf_matrix = confusion_matrix(ytest, y_pred)
    sns.heatmap(conf_matrix,annot=True,fmt='d')
    plt.xlabel("Predicted label") 
    plt.ylabel("True label") 
    plt.title("Confusion Matrix")

    print("Classification Report:\n", classification_report(ytest, y_pred))

    return model

def PlotFeatureImportance(X,catboost_model,gbm_model, rf_model):
    catboost_feature_importance = catboost_model.get_feature_importance()
    lightgbm_feature_importance = gbm_model.feature_importance()
    randomforest_feature_importance = rf_model.feature_importances_

        

    features = X.columns

    fi_df = pd.DataFrame({
        'feature': features,
        'catboost_importance': catboost_feature_importance,
        'lightgbm_importance': lightgbm_feature_importance,
        'randomforest_importance': randomforest_feature_importance
    })

    # --- normalize ---
    scaler = MinMaxScaler()
    fi_df[['catboost_importance','lightgbm_importance','randomforest_importance']] = scaler.fit_transform(
        fi_df[['catboost_importance','lightgbm_importance','randomforest_importance']]
    )

    # --- mean importance ---
    fi_df['mean_importance'] = fi_df[
        ['catboost_importance','lightgbm_importance','randomforest_importance']
    ].mean(axis=1)

    fi_df = fi_df.sort_values(by='mean_importance', ascending=False)

    # --- plotting: 2x2 subplots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    cols_to_plot = [
        'catboost_importance',
        'lightgbm_importance',
        'randomforest_importance',
        'mean_importance'
    ]

    titles = [
        'CatBoost Importance',
        'LightGBM Importance',
        'Random Forest Importance',
        'Mean Normalized Importance'
    ]

    for ax, col, title in zip(axes, cols_to_plot, titles):
        top5 = fi_df.nlargest(5, col)[['feature', col]]

        sns.barplot(
            data=top5,
            y='feature',   # horizontal bars = cleaner
            x=col,
            ax=ax,
            edgecolor='none'
        )

        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle("Top 5 Feature Importances by Model", fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return fi_df