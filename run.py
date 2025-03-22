import numpy as np

from exp_runner import run_exp
from optimizers import GridOptimizer, OptunaOptimizer


use_wandb = False
model_tp = 'catboost'
datasets = {
    'main': 'full',
#     'add': 500, # sample size
    'synt': 500, # synt data
}
kwargs = {"model_tp": model_tp, "datasets": datasets, "use_wandb": use_wandb}

# forest grid opt --> (0.04205295714325815, {'n_estimators': 140, 'max_depth': 15})
# args_to_grid = {
#     "n_estimators": np.arange(60, 160, 20),
#     "max_depth": np.arange(10, 35, 5),
# }

# forest optuna opt --> (0.04112548852422678, {'n_estimators': 297, 'max_depth': 26})
# args_to_tuna = {
#     "n_estimators": (int, 50, 500),
#     "max_depth": (int, 3, 30),
# }

# catboost grid opt --> (0.041153435706471064, {'iterations': 100, 'max_depth': 3})
# args_to_grid = {
#     "iterations": [50, 100, 200, 500],
#     "max_depth": [3, 5, 7, 9],
# }

# catboost optuna opt --> (0.03986117475773089, {'iterations': 499, 'max_depth': 6, 'learning_rate': 0.09604144154503975})
# args_to_tuna = {
#     "iterations": (int, 50, 500),
#     "max_depth": (int, 3, 11),
#     "learning_rate": (float, 1e-3, 1e-1),
# }

# opt = GridOptimizer(
#     args_to_grid=args_to_grid, **kwargs,
# )
# print(f"Best result: {opt.optimize()}")

# opt = OptunaOptimizer(
#     args_to_opt=args_to_tuna, **kwargs,
# )
# print(f"Best result: {opt.optimize(n_trials=25)}")

model_kwargs = {'iterations': 499, 'max_depth': 6, 'learning_rate': 0.09604144154503975}
print(f"Final metric: {run_exp(**kwargs, **model_kwargs)}")
