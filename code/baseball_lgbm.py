import sys; sys.path.append("../util")
from kazuyan_base import Timer, binary_metrics, fix_seed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgbm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, mean_squared_error
from sklearn.utils import compute_sample_weight

import shap
# 可視化のための javascript を読み込み
shap.initjs()

GBM_PARAMS = {
    "objective": "regression_l2", 
    "learning_rate": .005,
    # L2 Reguralization
    "reg_lambda": .1,
    # こちらは L1 
    "reg_alpha": .1,
    "max_depth": 7, 
    "n_estimators": 50000, 
    "colsample_bytree": .7, 
    "min_child_samples": 20,
    "subsample_freq": 3,
    "subsample": .9,
    # 特徴重要度計算のロジック(後述)
    "importance_type": "gain",
    "early_stopping_rounds": 50,
    "verbose_eval": 200,
    "metric": "rmse",
    "verbose": -1,
    "seed": SEED,
    'device_type': 'gpu',
}

def fit_lgbm(X: pd.DataFrame, y, cv, params):
    oof_pred = np.zeros(len(X), dtype=np.float32)
    models = []
    scores = []

    for i, (idx_train, idx_valid) in enumerate(cv):
        x_train, y_train = X[X.index.isin(idx_train)], y[idx_train]
        x_valid, y_valid = X[X.index.isin(idx_valid)], y[idx_valid]

        ds_train = lgbm.Dataset(x_train, y_train)
        ds_valid = lgbm.Dataset(x_valid, y_valid)

        with Timer(prefix=f"fit ========== Fold: {i + 1}"):
            model = lgbm.train(
                params,
                ds_train,
                valid_names=["train, valid"],
                valid_sets=[ds_train, ds_valid]
            )

            pred_i = model.predict(x_valid, num_iteration=model.best_iteration)
            oof_pred[idx_valid] = pred_i

            score = mean_squared_error(y_valid, pred_i)
            print(f" - fold{i + 1} - {score:.4f}")

            scores.append(score)
            models.append(model)
        
    score = mean_squared_error(y, oof_pred)
    print("=" * 50)
    print(f"FINISH: Whole Score: {score:.4f}")

    return oof_pred, models
fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv = fold.split(df_train, df_train["stadium_id@getNeedCol"])
oof, models = fit_lgbm(df_train, df_train_lab, cv, GBM_PARAMS)

def visualize_importance(models, feat_train_df, top_n):
    """lightGBM の model 配列の feature importance を plot する
    CVごとのブレを boxen plot として表現

    args:
        models:
            List of lightGBM models
        feat_train_df:
            学習時に使った DataFrame
    """
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["feature_importance"] = model.feature_importance(importance_type='split')
        _df["column"] = feat_train_df.columns
        _df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], 
                                          axis=0, ignore_index=True)

    order = feature_importance_df.groupby("column")\
        .sum()[["feature_importance"]]\
        .sort_values("feature_importance", ascending=False).index[:top_n]

    fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
    sns.boxenplot(data=feature_importance_df, 
                  x="feature_importance", 
                  y="column", 
                  order=order, 
                  ax=ax, 
                  palette="viridis", 
                  orient="h")
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Importance")
    ax.grid()
    fig.tight_layout()
    plt.show()
    return fig, ax

fig, ax = visualize_importance(models, df_train, 100)
