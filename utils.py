import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def plot_split_results(df, test_size, num_quantiles, seed):
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

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    names = [
        ['Initial Dataset', 'Train Dataset (Simple Split)', 'Test Dataset (Simple Split)'],
        ['Initial Dataset', 'Train Dataset (Quantile Split)', 'Test Dataset (Quantile Split)'],
    ]
    dfs = [
        [df, train_df, test_df], [df, train_df_quantile, test_df_quantile]
    ]
    for i in range(2):
        for j in range(3):
            axes[i, j].hist(dfs[i][j]['price'], bins=50, color='blue', alpha=0.7)
            axes[i, j].set_title(names[i][j])
            axes[i, j].set_xlabel('Price')
            axes[i, j].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    return train_df_quantile, test_df_quantile


def preprocess_data(df, num_features, cat_features):
    enc_cat = OneHotEncoder().fit_transform(df[cat_features]).toarray()
    return np.concatenate([df[num_features].to_numpy(), enc_cat], axis=1)
