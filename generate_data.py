import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from utils import preprocess_data


def fill_features(df, df_pre, df_add, df_add_pre, column_names, is_cat=True):
    df_add = df_add.copy()
    for col_name in column_names:
        if is_cat:
            enc = OneHotEncoder().fit(df[[col_name]])
            model = LinearRegression().fit(df_pre, enc.transform(df[[col_name]]).toarray())
            df_add[col_name] = enc.inverse_transform(model.predict(df_add_pre)).reshape(-1)
        else:
            model = LinearRegression().fit(df_pre, df[col_name])
            df_add[col_name] = np.clip(
                model.predict(df_add_pre).astype(int),
                a_min=df[col_name].min(),
                a_max=df[col_name].max()
            )
    return df_add


def generate_synt(df, num_features, cat_features, num_samples, tgt_name):
    df_gen = pd.DataFrame(columns=df.columns)
    for ft in num_features:
        df_gen[ft] = np.random.normal(
            loc=df[ft].mean(), scale=df[ft].std(), size=num_samples
        )
    for ft in cat_features:
        vals, counts = np.unique(df[ft], return_counts=True)
        df_gen[ft] = np.random.choice(
            vals, size=num_samples, p=np.array(counts) / sum(counts),
        )

    X_train = preprocess_data(df, num_features, cat_features)
    model = LinearRegression().fit(X_train, df[tgt_name])
    X_test = preprocess_data(df_gen, num_features, cat_features)
    df_gen[tgt_name] = model.predict(X_test).astype(int)

    return df_gen
