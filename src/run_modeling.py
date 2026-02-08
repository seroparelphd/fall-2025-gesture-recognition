#!/usr/bin/env python3
"""Compute-heavy modeling pipeline.

Trains models with within-user CV and writes model comparison artifacts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.base import clone
from xgboost import XGBClassifier

RANDOM_STATE = 13
PERSONALIZATION_K = 5


def run_personalization_cv(model, X, y, groups, name, k_folds):
    unique_users = groups.unique()
    all_f1_scores = []
    all_acc_scores = []
    per_user_results = []
    users_evaluated = 0
    users_skipped = 0

    cv_splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_STATE)

    for user in tqdm(unique_users, desc=f"Users ({name})"):
        user_indices = groups[groups == user].index
        X_user = X.loc[user_indices]
        y_user = y.loc[user_indices]

        class_counts = y_user.value_counts()
        if len(class_counts) < k_folds or class_counts.min() < k_folds:
            users_skipped += 1
            continue

        f1_scores = []
        acc_scores = []

        for train_idx, test_idx in cv_splitter.split(X_user, y_user):
            X_train = X_user.iloc[train_idx]
            X_test = X_user.iloc[test_idx]
            y_train = y_user.iloc[train_idx]
            y_test = y_user.iloc[test_idx]

            fold_le = LabelEncoder()
            y_train_enc = fold_le.fit_transform(y_train)
            y_test_enc = fold_le.transform(y_test)

            user_model = clone(model)
            user_model.fit(X_train, y_train_enc)
            y_pred = user_model.predict(X_test)

            f1_scores.append(f1_score(y_test_enc, y_pred, average='macro', zero_division=0))
            acc_scores.append(accuracy_score(y_test_enc, y_pred))

        all_f1_scores.extend(f1_scores)
        all_acc_scores.extend(acc_scores)
        users_evaluated += 1

        per_user_results.append({
            'Model': name,
            'User_ID': user,
            'Mean_User_Accuracy': float(np.mean(acc_scores)),
            'Mean_User_F1_Macro': float(np.mean(f1_scores)),
            'Sample_Count': int(len(X_user)),
            'Folds_Processed': int(len(acc_scores))
        })

    mean_f1 = float(np.mean(all_f1_scores))
    std_f1 = float(np.std(all_f1_scores))
    mean_acc = float(np.mean(all_acc_scores))
    std_acc = float(np.std(all_acc_scores))

    aggregate_result = {
        'Model': name,
        'Mean_Accuracy': mean_acc,
        'Std_Accuracy': std_acc,
        'Mean_F1_Macro': mean_f1,
        'Std_F1_Macro': std_f1,
        'Hyperparameters': str(model.named_steps['clf'].get_params(deep=False)),
        'N_Users_Evaluated': users_evaluated,
        'N_Users_Skipped': users_skipped,
        'Total_Folds': len(all_f1_scores),
    }

    return aggregate_result, pd.DataFrame(per_user_results)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "processed"
    results_dir = root / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_selected = data_dir / "train_calib_selected.csv"
    train_full = data_dir / "train_calib_full.csv"

    if not train_selected.exists() or not train_full.exists():
        raise FileNotFoundError("Missing training data outputs from feature selection step.")

    print("🚀 Loading training data...")
    df_train = pd.read_csv(train_selected)
    df_train_full = pd.read_csv(train_full)

    X = df_train.drop(columns=['user', 'gesture', 'stage'])
    y = df_train['gesture']
    groups = df_train['user']

    X_full = df_train_full.drop(columns=['user', 'gesture', 'stage'])

    preprocessor = Pipeline([('log_transform', FunctionTransformer(np.log1p)), ('scaler', StandardScaler())])

    models_to_test = {
        'DummyClassifier': Pipeline([
            ('prep', preprocessor),
            ('clf', DummyClassifier(strategy='stratified', random_state=RANDOM_STATE))
        ]),
        'Logit_L2': Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(
                solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE
            ))
        ]),
        'RandomForest': Pipeline([
            ('prep', preprocessor),
            ('clf', RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=RANDOM_STATE, n_jobs=1,
                class_weight='balanced'
            ))
        ]),
        'DecisionTree': Pipeline([
            ('prep', preprocessor),
            ('clf', DecisionTreeClassifier(
                max_depth=10, random_state=RANDOM_STATE,
                class_weight='balanced'
            ))
        ]),
        'Logit_Weighted_L2': Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(
                solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE,
                class_weight='balanced'
            ))
        ]),
        'Logit_Weighted_All_L2': Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(
                solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE,
                class_weight='balanced', C=0.1
            ))
        ]),
        'Logit_All_L2': Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(
                solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE, C=0.1
            ))
        ]),
        'XGBoost': Pipeline([
            ('prep', preprocessor),
            ('clf', XGBClassifier(
                n_estimators=100, max_depth=6,
                learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                objective='multi:softprob', eval_metric='mlogloss',
                random_state=RANDOM_STATE, n_jobs=1
            ))
        ])
    }

    model_feature_mapping = {
        'Logit_All_L2': X_full,
        'Logit_Weighted_All_L2': X_full,
    }

    results = []
    all_user_results = []

    for name, model in tqdm(models_to_test.items(), desc="Models"):
        if name in model_feature_mapping:
            X_model = model_feature_mapping[name]
            feature_type = "full features"
        else:
            X_model = X
            feature_type = "selected features"
        print(f"🧠 Evaluating {name} with {feature_type} ({X_model.shape[1]} features)")
        res, user_res = run_personalization_cv(model, X_model, y, groups, name, PERSONALIZATION_K)
        results.append(res)
        all_user_results.append(user_res)
        print(f"✅ {name} complete")

    user_performance_df = pd.concat(all_user_results, ignore_index=True)
    results_df = pd.DataFrame(results).sort_values(by='Mean_F1_Macro', ascending=False)

    results_path = results_dir / "model_comparison.csv"
    results_df['experiment_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
    results_df['experiment_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    results_df.to_csv(results_path, index=False)

    user_perf_path = results_dir / "model_user_performance.csv"
    user_performance_df.to_csv(user_perf_path, index=False)

    print(f"📈 Model comparison saved to {results_path}")
    print(f"📊 User performance saved to {user_perf_path}")


if __name__ == "__main__":
    main()
