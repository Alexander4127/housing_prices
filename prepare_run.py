from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC

from load_data import load_main_data, load_add_data, fill_features, generate_synt
from utils import preprocess_data, split_results, MeanModel


def setup_model(model_tp, verbose, seed, **kwargs):
    model_types = ['mean', 'ridge', 'svm', 'tree', 'forest', 'catboost']
    assert model_tp in model_types, f'Invalid model type, expected one of {model_types}'
    if model_tp == 'mean':
        model = MeanModel(**kwargs)
    elif model_tp == 'ridge':
        model = Ridge(random_state=seed, **kwargs)
    elif model_tp == 'svm':
        model = SVC(random_state=seed, **kwargs)
    elif model_tp == 'tree':
        model = DecisionTreeRegressor(random_state=seed, **kwargs)
    elif model_tp == 'forest':
        model = RandomForestRegressor(verbose=verbose, random_state=seed, **kwargs)
    elif model_tp == 'catboost':
        model = CatBoostRegressor(verbose=verbose, random_seed=seed, **kwargs)
    return model


def setup_data(datasets, verbose, seed):
    df, num_features, cat_features = load_main_data(verbose=verbose)
    df, df_test = split_results(df, test_size=0.2, num_quantiles=10, seed=seed)
    if datasets['main'] != 'full':
        df = df.sample(n=datasets['main'])

    data = [df]
    if 'add' in datasets:
        df_add, num_features_add, cat_features_add = load_add_data(
            df, num_features, cat_features, verbose=verbose
        )
        if datasets['add'] != 'full':
            df_add = df_add.sample(n=datasets['add'])
        df_pre = preprocess_data(df, ["price"] + num_features_add, cat_features_add)
        df_add_pre = preprocess_data(df_add, ["price"] + num_features_add, cat_features_add)

        df_add = fill_features(
            df, df_pre, df_add, df_add_pre,
            list(set(cat_features) - set(cat_features_add))
        )
        df_add = fill_features(
            df, df_pre, df_add, df_add_pre,
            list(set(num_features) - set(num_features_add)), is_cat=False
        )
        data.append(df_add)

    if 'synt' in datasets:
        np.random.seed(seed)
        df_synt = generate_synt(df, num_features, cat_features, num_samples=5000, tgt_name='price')
        if datasets['synt'] != 'full':
            df_synt = df_synt.sample(n=datasets['synt'])
        data.append(df_synt)

    data = pd.concat(data)
    if verbose:
        print(f"Data shape: {data.shape}")

    X_train = preprocess_data(data, num_features, cat_features)
    y_train = data['price']

    X_test = preprocess_data(df_test, num_features, cat_features)
    y_test = df_test['price']

    return X_train, y_train, X_test, y_test
