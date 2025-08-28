import pandas as pd
from pandas import IndexSlice as idx
import numpy as np

from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score)

def split_data(df):
    y = df.filter(like='target')
    X = df.drop(y.columns, axis=1)

    return X, y

def clean_xy(X, y):
    valid_index = X.dropna().index.intersection(y.dropna().index)
    # 过滤 X 和 y
    X_clean = X.loc[valid_index].copy()
    y_clean = y.loc[valid_index].copy()

    return X_clean, y_clean

def fit_predict_last_line(model, X, y, train_length=24*7*4):

    X_train = X.iloc[-1 - train_length:-1]
    y_train = y[-1 - train_length:-1]
    X_pred = X.iloc[-1:]
    # y_test = y[-1:]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_pred)

    ts = X_pred.index[-1]

    return {"timestamp": ts,
            "y_pred": float(y_pred[-1])}

def fit_predict_regression_model(model, X, y, cv):
    """
    用时间序列交叉验证训练模型，收集预测值，并按日期计算每日IC和RMSE。

    参数:
    - model: 拟合的回归模型（需有 fit 和 predict 方法）
    - X: 特征数据，pandas DataFrame
    - y: 目标变量，pandas Series
    - cv: sklearn.model_selection 中的 cross-validator，例如 TimeSeriesSplit

    返回:
    - predictions_df: 包含 actuals 和 predicted 的 DataFrame，按时间索引
    """
    all_predictions = []

    for train_idx, test_idx in cv.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        preds = pd.DataFrame({
            'actuals': y_test,
            'predicted': y_pred
        }, index=X_test.index)

        all_predictions.append(preds)

    predictions_df = pd.concat(all_predictions).sort_index()

    return predictions_df

def score_reg_pred(pred_df):
    """按天计算 IC 和 RMSE"""
    preds_by_day = pred_df.groupby(pred_df.index.date)
    ic_series = preds_by_day.apply(lambda x: spearmanr(x.predicted, x.actuals)[0] * 100)
    rmse_series = preds_by_day.apply(lambda x: np.sqrt(mean_squared_error(x.actuals, x.predicted)))

    score_df = pd.concat([ic_series.to_frame('ic'), rmse_series.to_frame('rmse')], axis=1)

    return score_df


def fit_predict_classification_model(model, X, y, cv):
    """
    对分类模型进行时间序列交叉验证训练和预测，并按 index.date 计算每日 accuracy, precision, recall, f1, roc_auc

    参数:
    - model: 拟合的 sklearn 分类模型，例如 LogisticRegression()
    - X: 特征 DataFrame，index 为 datetime
    - y: 标签 Series，为二分类
    - cv: TimeSeriesSplit 交叉验证器

    返回:
    - all_predictions: 包含 actuals, predicted_label, predicted_proba 的 DataFrame
    """
    all_predictions = []

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # 二分类时取第1类概率

        preds = pd.DataFrame({
            'actuals': y_test,
            'predicted_label': y_pred,
            'predicted_proba': y_proba
        }, index=X_test.index)
        all_predictions.append(preds)

    all_predictions = pd.concat(all_predictions).sort_index()

    return all_predictions

def score_clf_pred(pred_df):
    """按天计算 accuracy/precision/recall/f1/roc_auc/ic"""
    grouped = pred_df.groupby(pred_df.index.date)

    daily_scores = pd.DataFrame({
        'accuracy': grouped.apply(lambda x: accuracy_score(x.actuals, x.predicted_label)),
        'precision': grouped.apply(lambda x: precision_score(x.actuals, x.predicted_label, zero_division=0)),
        'recall': grouped.apply(lambda x: recall_score(x.actuals, x.predicted_label, zero_division=0)),
        'f1': grouped.apply(lambda x: f1_score(x.actuals, x.predicted_label, zero_division=0)),
        'roc_auc': grouped.apply(
            lambda x: roc_auc_score(x.actuals, x.predicted_proba) if len(x.actuals.unique()) == 2 else np.nan),
        'ic': grouped.apply(lambda x: spearmanr(x.predicted_proba, x.actuals)[0] * 100)
    })

    # 统一索引为 datetime 格式
    daily_scores.index = pd.to_datetime(daily_scores.index)

    return daily_scores

if __name__ == '__main__':
    from timeseries_cv import MultipleTimeSeriesCV

    import sqlite3
    import pandas as pd

    price_all = pd.read_sql_query(
        f"SELECT * FROM kline ORDER BY datetime",
        sqlite3.connect('../data/crypto_data.db'),
        parse_dates=['datetime']
    )

    price_all.set_index(['symbol', 'datetime'], inplace=True)

    print(price_all.index.get_level_values('datetime').max())

    train_period_length = 24 * 7 * 4
    test_period_length = 24
    lookahead = 1

    cv = MultipleTimeSeriesCV(
        train_length=train_period_length,
        test_length=test_period_length,
        lookahead=lookahead)

    for train_idx, test_idx in cv.split(price_all):
        X_train = price_all.iloc[train_idx]
        X_test = price_all.iloc[test_idx]

        print(f"test period: {X_test.index.get_level_values('datetime').min()}, {X_test.index.get_level_values('datetime').max()}")
        print(f"train period: {X_train.index.get_level_values('datetime').min()}, {X_train.index.get_level_values('datetime').max()}")


