import kagglehub
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from .utils import preprocess_data


def print_stats(df, names, desc):
    print(desc)
    df_cat = pd.DataFrame(columns=["Name", "Vals", "Counts"])
    for ft in names:
        df_cat.loc[len(df_cat)] = [ft] + list(map(list, np.unique(df[ft], return_counts=True)))
    print(df_cat.to_markdown(), end="\n\n")


def load_main_data(verbose=True):
    # divide features to num and cat
    # here we move ordered cat features to numerical ones
    num_features = ['area', 'logarea', 'bathrooms', 'bedrooms', 'stories', 'parking']
    cat_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
    df = pd.read_csv(f"{path}/Housing.csv")

    # drop duplicates and "similar" objects
    df = df.drop_duplicates(subset=["area", "price"])

    assert df.isna().sum().sum() == 0
    assert ((df["price"] > 0) & (df.area > 0)).all()

    df["logarea"] = np.log(df.area)
    assert len(num_features) + len(cat_features) + 1 == len(df.columns)

    if verbose:
        print_stats(df, cat_features, "Cat features distributions")

    if verbose:
        print_stats(
            df, num_features[2:],
            "Cat features converted to numerical (avoid overfitting on rare samples)"
        )

    if verbose:
        print("DataFrame:")

    return df, num_features, cat_features

def load_add_data(main_df, main_num_features, main_cat_features, verbose=True):
    path_add = kagglehub.dataset_download("sukhmandeepsinghbrar/housing-price-dataset")

    rename_columns = {
        "bedrooms": "bedrooms",
        "bathrooms": "bathrooms",
        "sqft_lot": "area",
        "floors": "stories",
        "price": "price",
        "logarea": "logarea",
    }

    num_features = ["area", "logarea", "bedrooms", "bathrooms", "stories"]
    cat_features = []

    assert sorted(num_features) == sorted(set(main_num_features) & set(rename_columns.values()))
    assert sorted(cat_features) == sorted(set(main_cat_features) & set(rename_columns.values()))

    df = pd.read_csv(f"{path_add}/Housing.csv")
    df["logarea"] = np.log(df["sqft_lot"])

    # drop duplicates
    if len(df['id'].unique()) != len(df):
        if verbose:
            print(f"Find duplicates. Dropping them...")
        df = df.drop_duplicates(subset=["id"])
        assert len(df['id'].unique()) == len(df)

    # keep only common columns with the initial dataframe
    df = df[list(rename_columns.keys())].rename(columns=rename_columns).astype(int)

    if verbose:
        print_stats(df, num_features[2:], "Numerical features before removing extra values")

    # remove cat features with extra values compared to orig df
    for cat_feat in num_features[2:]:
        orig_vals = main_df[cat_feat].unique()
        df = df[df[cat_feat].isin(orig_vals)]

    if verbose:
        print_stats(df, num_features[2:], "Final numerical features")

    # check nans and incorrect samples
    assert df.isna().sum().sum() == 0
    assert ((df["price"] > 0) & (df.area > 0)).all()

    if verbose:
        print("Additional DataFrame")

    return df, num_features, cat_features


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
