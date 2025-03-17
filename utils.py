import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def split_results(df, test_size, num_quantiles, seed, verbose=True):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)

    quantiles = pd.qcut(df['price'], q=num_quantiles, labels=False)
    train_dfs = []
    test_dfs = []

    for q in range(num_quantiles):
        q_df = df[quantiles == q]
        q_train, q_test = train_test_split(q_df, test_size=test_size, random_state=seed)
        train_dfs.append(q_train)
        test_dfs.append(q_test)

    train_df_quantile = pd.concat(train_dfs)
    test_df_quantile = pd.concat(test_dfs)

    if verbose:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        names = [
            ['Initial Dataset', 'Train Dataset (Simple Split)', 'Test Dataset (Simple Split)'],
            ['Initial Dataset', 'Train Dataset (Quantile Split)', 'Test Dataset (Quantile Split)'],
        ]
        dfs = [
            [df, train_df, test_df], [df, train_df_quantile, test_df_quantile]
        ]
        colors = ['blue', 'green', 'red']
        for i in range(2):
            for j in range(3):
                axes[i, j].hist(dfs[i][j]['price'], bins=50, color=colors[j], alpha=0.7)
                axes[i, j].set_title(names[i][j])
                axes[i, j].set_xlabel('Price')
                axes[i, j].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    return train_df_quantile, test_df_quantile


def preprocess_data(df, num_features, cat_features):
    enc_cat = OneHotEncoder().fit_transform(df[cat_features]).toarray()
    return np.concatenate([df[num_features].to_numpy(), enc_cat], axis=1)


def calc_metrics(y_true, y_pred):
    return {
        "RelMSE": mean_squared_error(np.ones_like(y_true), y_pred / y_true),
        "RelMAE": mean_absolute_error(np.ones_like(y_true), y_pred / y_true),
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


class MeanModel:
    def __init__(self):
        self.pred = None

    def fit(self, X_train, y_train):
        self.pred = y_train.mean()

    def predict(self, X_test):
        return np.ones(len(X_test)) * self.pred
