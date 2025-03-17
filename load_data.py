import kagglehub
import pandas as pd
import numpy as np


def print_stats(df, names, desc):
    print(desc)
    df_cat = pd.DataFrame(columns=["Name", "Vals", "Counts"])
    for ft in names:
        df_cat.loc[len(df_cat)] = [ft] + list(np.unique(df[ft], return_counts=True))
    print(df_cat.to_markdown(), end="\n\n")


def load_main_data():
    # divide features to num and cat
    # here we move ordered cat features to numerical ones
    num_features = ['area', 'bathrooms', 'bedrooms', 'stories', 'parking']
    cat_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
    df = pd.read_csv(f"{path}/Housing.csv")

    assert len(num_features) + len(cat_features) + 1 == len(df.columns)

    # drop duplicates and "similar" objects
    df = df.drop_duplicates(subset=["area", "price"])

    assert df.isna().sum().sum() == 0
    assert ((df["price"] > 0) & (df.area > 0)).all()

    print_stats(df, cat_features, "Cat features distributions")

    print_stats(
        df, num_features[1:],
        "Cat features converted to numerical (avoid overfitting on rare samples)"
    )

    print("DataFrame:")
    print(df.head(3).to_markdown())

    return df, num_features, cat_features

def load_add_data(main_df, main_num_features, main_cat_features):
    path_add = kagglehub.dataset_download("sukhmandeepsinghbrar/housing-price-dataset")

    rename_columns = {
        "bedrooms": "bedrooms",
        "bathrooms": "bathrooms",
        "sqft_lot": "area",
        "floors": "stories",
        "price": "price",
    }

    num_features = ["area", "bedrooms", "bathrooms", "stories"]
    cat_features = []

    assert sorted(num_features) == sorted(set(main_num_features) & set(rename_columns.values()))
    assert sorted(cat_features) == sorted(set(main_cat_features) & set(rename_columns.values()))

    df = pd.read_csv(f"{path_add}/Housing.csv")

    # drop duplicates
    if len(df['id'].unique()) != len(df):
        print(f"Find duplicates. Dropping them...")
        df = df.drop_duplicates(subset=["id"])
        assert len(df['id'].unique()) == len(df)

    # keep only common columns with the initial dataframe
    df = df[list(rename_columns.keys())].rename(columns=rename_columns).astype(int)

    print_stats(df, num_features[1:], "Numerical features before removing extra values")

    # remove cat features with extra values compared to orig df
    for cat_feat in num_features[1:]:
        orig_vals = main_df[cat_feat].unique()
        df = df[df[cat_feat].isin(orig_vals)]

    print_stats(df, num_features[1:], "Final numerical features")

    # check nans and incorrect samples
    assert df.isna().sum().sum() == 0
    assert ((df["price"] > 0) & (df.area > 0)).all()

    print("Additional DataFrame")
    print(df.head(3).to_markdown())

    return df, num_features, cat_features
