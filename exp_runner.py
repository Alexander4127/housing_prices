import pandas as pd
import wandb

from utils import calc_metrics
from prepare_run import setup_model, setup_data


def run_exp(
    model_tp,
    datasets,
    verbose=False,
    seed=42,
    use_wandb=False,
    main_metric="RelMSE",
    **model_kwargs,
) -> float:
    if use_wandb:
        data_str = "_".join(f"{k}:{v}" for k, v in datasets.items())
        args_str = ("_args_" + "_".join(
            f"{name}:{val}" for name, val in sorted(model_kwargs.items())
        )) if model_kwargs else ""
        wandb.init(
            project="housing-pricing",
            name=f"model_{model_tp}{args_str}_data_{data_str}"
        )

    model = setup_model(model_tp, verbose, seed, **model_kwargs)
    X_train, y_train, X_test, y_test = setup_data(datasets, verbose, seed)

    model.fit(X_train, y_train)

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

    return test_metrics[main_metric]
