#!/usr/bin/env python3
"""Compute-heavy modeling pipeline.

Trains models with within-user CV and writes model comparison artifacts.
"""
from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
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
SELECTOR_MAX_FEATURES = 37


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
    data_dir = Path(os.environ.get("PIPELINE_DATA_DIR", root / "data" / "processed"))
    results_dir = Path(os.environ.get("PIPELINE_RESULTS_DIR", root / "results" / "tables"))
    results_dir.mkdir(parents=True, exist_ok=True)

    train_full = data_dir / "train_calib_full.csv"

    if not train_full.exists():
        raise FileNotFoundError("Missing training data outputs from feature selection step.")

    print("🚀 Loading training data...")
    df_train_full = pd.read_csv(train_full)

    X_full = df_train_full.drop(columns=['user', 'gesture', 'stage'])
    y = df_train_full['gesture']
    groups = df_train_full['user']

    selector_estimator = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    def build_pipeline(clf, use_selector: bool) -> Pipeline:
        steps = [
            ('log_transform', FunctionTransformer(np.log1p)),
            ('scaler', StandardScaler()),
        ]
        if use_selector:
            steps.append((
                'selector',
                SelectFromModel(
                    selector_estimator,
                    max_features=SELECTOR_MAX_FEATURES,
                    threshold=-np.inf
                )
            ))
        steps.append(('clf', clf))
        return Pipeline(steps)

    models_to_test = {
        'DummyClassifier': DummyClassifier(strategy='stratified', random_state=RANDOM_STATE),
        'Logit_L2': LogisticRegression(
            solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=RANDOM_STATE, n_jobs=1,
            class_weight='balanced'
        ),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'Logit_Weighted_L2': LogisticRegression(
            solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'Logit_Weighted_All_L2': LogisticRegression(
            solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE,
            class_weight='balanced', C=0.1
        ),
        'Logit_All_L2': LogisticRegression(
            solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE, C=0.1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
            objective='multi:softprob', eval_metric='mlogloss',
            random_state=RANDOM_STATE, n_jobs=1
        )
    }

    full_feature_models = {'Logit_All_L2', 'Logit_Weighted_All_L2'}
    if os.environ.get("PIPELINE_SKIP_XGBOOST") == "1":
        models_to_test.pop('XGBoost', None)

    results = []
    all_user_results = []

    for name, clf in tqdm(models_to_test.items(), desc="Models"):
        use_selector = name not in full_feature_models
        feature_type = "full features" if not use_selector else "selected features (fold-safe)"
        model = build_pipeline(clf, use_selector)
        print(f"🧠 Evaluating {name} with {feature_type} ({X_full.shape[1]} features)")
        res, user_res = run_personalization_cv(model, X_full, y, groups, name, PERSONALIZATION_K)
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
