#!/usr/bin/env python3
"""Compute-heavy feature selection pipeline.

Reads cleaned data, performs per-user split, computes feature importances,
redundancy pruning, and writes CSV artifacts used by reporting notebooks.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "processed"
    results_dir = root / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    cleaned_data_path = data_dir / "features_emg_data_cleaned.csv"
    if not cleaned_data_path.exists():
        raise FileNotFoundError(f"Missing cleaned data: {cleaned_data_path}")

    print("🚀 Loading cleaned data...")
    df_no_outliers = pd.read_csv(cleaned_data_path)

    metadata_cols = ['user', 'gesture', 'stage', 'is_outlier']
    all_columns = df_no_outliers.columns.tolist()
    feature_cols = [col for col in all_columns if col not in metadata_cols]

    print("🔪 Performing personalization split (train/calibration vs test) with group-aware splitting...")
    df_no_outliers['stage_gesture'] = df_no_outliers['stage'] + '__' + df_no_outliers['gesture']

    group_candidates = [
        'trial_id', 'trial', 'repetition', 'rep', 'prompt_idx', 'prompt_index',
        'trial_index', 'event_index', 'timestamp', 'time', 'start', 'end'
    ]
    group_col = next((c for c in group_candidates if c in df_no_outliers.columns), None)
    if group_col is None:
        raise ValueError(
            "No trial/repetition identifier found in cleaned data. "
            "Add a trial-level column (e.g., 'trial_id' or 'prompt_idx') in feature extraction "
            "so group-based splitting can prevent temporal leakage."
        )

    df_no_outliers['group_id'] = (
        df_no_outliers['user'].astype(str)
        + '__' + df_no_outliers['stage_gesture'].astype(str)
        + '__' + df_no_outliers[group_col].astype(str)
    )

    train_pieces, test_pieces = [], []
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=13)

    for user_id in tqdm(df_no_outliers['user'].unique(), desc="Split users (group-aware)"):
        user_data = df_no_outliers[df_no_outliers['user'] == user_id]
        for stage_gesture in user_data['stage_gesture'].unique():
            subset = user_data[user_data['stage_gesture'] == stage_gesture]
            groups = subset['group_id']
            n_groups = groups.nunique()

            if n_groups <= 1:
                train_pieces.append(subset)
                continue

            if n_groups < 5:
                test_group = groups.unique()[0]
                train_pieces.append(subset[subset['group_id'] != test_group])
                test_pieces.append(subset[subset['group_id'] == test_group])
                continue

            train_idx, test_idx = next(splitter.split(subset, groups=groups))
            train_pieces.append(subset.iloc[train_idx])
            test_pieces.append(subset.iloc[test_idx])

    df_train = pd.concat(train_pieces)
    df_test = pd.concat(test_pieces)

    df_no_outliers = df_no_outliers.drop(columns=['stage_gesture', 'group_id'])
    df_train = df_train.drop(columns=['stage_gesture', 'group_id'])
    df_test = df_test.drop(columns=['stage_gesture', 'group_id'])

    print(f"📊 Split complete. Train: {df_train.shape}, Test: {df_test.shape}")

    # Save full feature sets for modeling
    print("💾 Saving full feature sets (holdout saved once, never used for selection/CV)...")
    train_full = df_train[['user', 'gesture', 'stage'] + feature_cols].copy()
    test_full = df_test[['user', 'gesture', 'stage'] + feature_cols].copy()
    train_full.to_csv(data_dir / "train_calib_full.csv", index=False)
    test_full.to_csv(data_dir / "test_holdout_full.csv", index=False)

    # Preprocess (log + scale)
    preprocessing_pipeline = Pipeline([('log_transform', FunctionTransformer(np.log1p)), ('scaler', StandardScaler())])
    X_train_processed = preprocessing_pipeline.fit_transform(df_train[feature_cols])
    X_test_processed = preprocessing_pipeline.transform(df_test[feature_cols])

    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_cols, index=df_train.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_cols, index=df_test.index)

    # Feature importance
    importance_file = data_dir / "feature_importance_ranking.csv"
    if importance_file.exists():
        print(f"⚠️ Removing cached feature importance: {importance_file}")
        importance_file.unlink()

    print("🧠 Training RandomForest for feature importance...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=13,
        n_jobs=-1,
    )
    rf_model.fit(X_train_processed, df_train['gesture'])
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_,
    }).sort_values('importance', ascending=False)
    feature_importance_df.to_csv(importance_file, index=False)
    print("✅ Feature importance saved.")

    # Per-user importances
    print("📉 Computing per-user feature importance stats...")
    all_user_results = {}
    for user_id in tqdm(df_train['user'].unique(), desc="User RF importance"):
        user_data = df_train[df_train['user'] == user_id]
        X_user = user_data[feature_cols].copy()
        y_user = user_data['gesture']
        rf_user = RandomForestClassifier(
            n_estimators=100,
            random_state=13,
            n_jobs=-1,
        )
        rf_user.fit(X_user, y_user)
        user_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_user.feature_importances_,
        }).sort_values('importance', ascending=False)
        all_user_results[user_id] = user_importance_df

    feature_importance_stats = {}
    for feature in feature_cols:
        scores = []
        for _, user_imp_df in all_user_results.items():
            s = user_imp_df.loc[user_imp_df['feature'] == feature, 'importance']
            if len(s) > 0:
                scores.append(float(s.iloc[0]))
        if len(scores) > 0:
            scores = np.asarray(scores, dtype=float)
            mean_val = float(np.mean(scores))
            median_val = float(np.median(scores))
            std_val = float(np.std(scores, ddof=0))
            n_scored = int(len(scores))
        else:
            mean_val = 0.0
            median_val = 0.0
            std_val = 0.0
            n_scored = 0
        feature_importance_stats[feature] = {
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'n_users_with_score': n_scored,
        }

    per_user_stats_df = pd.DataFrame.from_dict(feature_importance_stats, orient='index')

    # Correlation matrix for pruning
    STARTING_POOL_SIZE = 87
    CORRELATION_THRESHOLD = 0.875
    topN_features = feature_importance_df.head(STARTING_POOL_SIZE)['feature'].tolist()
    corr_df = df_train[topN_features].corr()

    # Redundancy pruning
    stats_tbl = per_user_stats_df.copy()
    common_idx = stats_tbl.index.intersection(corr_df.index)
    stats_tbl = stats_tbl.loc[common_idx].copy()
    rank_col = 'median' if 'median' in stats_tbl.columns else 'mean'
    top_features = (
        stats_tbl.sort_values(rank_col, ascending=False)
        .head(int(STARTING_POOL_SIZE))
        .index.tolist()
    )

    final_features = []
    for f in top_features:
        keep_f = True
        for g in final_features:
            if (f in corr_df.index and g in corr_df.index and f in corr_df.columns and g in corr_df.columns):
                rho = float(abs(corr_df.loc[f, g]))
            else:
                rho = 0.0
            if rho > float(CORRELATION_THRESHOLD):
                if float(stats_tbl.loc[f, rank_col]) > float(stats_tbl.loc[g, rank_col]):
                    final_features.remove(g)
                    final_features.append(f)
                keep_f = False
                break
        if keep_f and f not in final_features:
            final_features.append(f)

    # Build feature selection log
    if 'median' in per_user_stats_df.columns:
        max_median = per_user_stats_df['median'].max()
        per_user_stats_df['importance_norm_median'] = per_user_stats_df['median'] / max_median
    else:
        per_user_stats_df['importance_norm_median'] = np.nan

    if 'mean' in per_user_stats_df.columns:
        max_mean = per_user_stats_df['mean'].max()
        per_user_stats_df['importance_norm_mean'] = per_user_stats_df['mean'] / max_mean
    else:
        per_user_stats_df['importance_norm_mean'] = np.nan

    sorted_df = per_user_stats_df.sort_values(rank_col, ascending=False).copy()
    ranked_features = list(sorted_df.index)
    feature_to_rank = {fname: i for i, fname in enumerate(ranked_features, start=1)}

    seed_top = ranked_features[:int(STARTING_POOL_SIZE)]
    seed_top_set = set(seed_top)
    final_set = set(final_features)

    rows = []
    for feat in feature_cols:
        if feat in per_user_stats_df.index:
            imp_median = float(per_user_stats_df.loc[feat, 'median']) if 'median' in per_user_stats_df.columns else np.nan
            imp_mean = float(per_user_stats_df.loc[feat, 'mean']) if 'mean' in per_user_stats_df.columns else np.nan
            norm_median = float(per_user_stats_df.loc[feat, 'importance_norm_median']) if 'importance_norm_median' in per_user_stats_df.columns else np.nan
            norm_mean = float(per_user_stats_df.loc[feat, 'importance_norm_mean']) if 'importance_norm_mean' in per_user_stats_df.columns else np.nan
        else:
            imp_median = np.nan
            imp_mean = np.nan
            norm_median = np.nan
            norm_mean = np.nan

        rank_val = int(feature_to_rank[feat]) if feat in feature_to_rank else np.nan

        channel = feat.split('_')[0] if '_' in feat else 'Unknown'
        lower_name = feat.lower()
        if 'fft' in lower_name:
            category = 'FFT'
        elif 'rms' in lower_name:
            category = 'RMS'
        elif 'mav' in lower_name:
            category = 'MAV'
        elif 'max' in lower_name:
            category = 'Max'
        elif 'thresh' in lower_name:
            category = 'Thresh'
        else:
            category = 'Other'

        if feat in final_set:
            decision = 'KEEP'
            replaced_by = ''
        else:
            decision = 'DROP'
            replaced_by = ''
            if feat in seed_top_set:
                best_r = -np.inf
                best_feat = ''
                for kept in final_features:
                    if (feat in corr_df.index) and (kept in corr_df.columns):
                        r_val = corr_df.loc[feat, kept]
                        if pd.notna(r_val):
                            a = abs(float(r_val))
                            if a > best_r:
                                best_r = a
                                best_feat = kept
                if np.isfinite(best_r) and (best_r > float(CORRELATION_THRESHOLD)):
                    replaced_by = best_feat

        rows.append({
            'feature': feat,
            'channel': channel,
            'category': category,
            'importance_median': imp_median,
            'importance_mean': imp_mean,
            'importance_norm_median': norm_median,
            'importance_norm_mean': norm_mean,
            'rank': rank_val,
            'in_top_n': 1 if feat in seed_top_set else 0,
            'in_final_set': 1 if feat in final_set else 0,
            'decision': decision,
            'replaced_by': replaced_by,
        })

    feature_selection_df = pd.DataFrame(rows).sort_values(['decision', 'rank'])
    feat_path = results_dir / "feature_selection.csv"
    feature_selection_df.to_csv(feat_path, index=False)

    # Save selected train/test sets
    cols_keep = ['user', 'gesture', 'stage'] + list(final_features)
    train_out = df_train[cols_keep].copy()
    test_out = df_test[cols_keep].copy()

    train_out.to_csv(data_dir / "train_calib_selected.csv", index=False)
    test_out.to_csv(data_dir / "test_holdout_selected.csv", index=False)

    print(f"✅ Features reduced to {len(final_features)}. Saved selection to {feat_path}")


if __name__ == "__main__":
    main()
