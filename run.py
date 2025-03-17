import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import wandb

from generate_data import fill_features, generate_synt
from load_data import load_main_data, load_add_data
from utils import calc_metrics, preprocess_data, split_results, MeanModel

verbose = False
use_wandb = True
model_tp = 'mean'
datasets = {
    'main': 'full',
#     'add': 500, # sample size
#     'synt': 500, # synt data
}

if use_wandb:
    data_str = "_".join(f"{k}:{v}" for k, v in datasets.items())
    wandb.init(
        project="housing-pricing",
        name=f"model_{model_tp}_data_{data_str}"
    )

if model_tp == 'mean':
    model = MeanModel()
elif model_tp == 'ridge':
    model = Ridge()
else:
    raise RuntimeError(f"Unexpected model name {model_tp}")

df, num_features, cat_features = load_main_data(verbose=verbose)
df, df_test = split_results(df, test_size=0.2, num_quantiles=10, seed=42)
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
    df_synt = generate_synt(df, num_features, cat_features, num_samples=5000, tgt_name='price')
    if datasets['synt'] != 'full':
        df_synt = df_synt.sample(n=datasets['synt'])
    data.append(df_synt)


data = pd.concat(data)
if verbose:
    print(f"Data shape: {data.shape}")

X_train = preprocess_data(data, num_features, cat_features)
y_train = data['price']
model.fit(X_train, y_train)

X_test = preprocess_data(df_test, num_features, cat_features)
y_test = df_test['price']

train_metrics = calc_metrics(y_train, model.predict(X_train))
test_metrics = calc_metrics(y_test, model.predict(X_test))

df_res = pd.DataFrame(columns=train_metrics.keys())
for k, v in train_metrics.items():
    df_res.loc["Train", k] = v
for k, v in test_metrics.items():
    df_res.loc["Test", k] = v

print(df_res)

if use_wandb:
    wandb.log({"train_" + k: v for k, v in train_metrics.items()})
    wandb.log({"test_" + k: v for k, v in test_metrics.items()})
    wandb.finish()
