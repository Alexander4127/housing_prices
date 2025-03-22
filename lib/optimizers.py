import itertools
import numpy as np
import optuna

from .exp_runner import run_exp

class GridOptimizer:
    def __init__(self, args_to_grid, **kwargs):
        self.args_to_grid = args_to_grid
        self.kwargs = kwargs

    def optimize(self):
        best_metric, best_vals = np.inf, None
        keys, val_ranges = zip(*self.args_to_grid.items())
        for vals in itertools.product(*val_ranges):
            dict_vals = dict(zip(keys, vals))
            print(f"Running exp for {dict_vals}")
            cur_metric = run_exp(**self.kwargs, **dict_vals)
            if cur_metric < best_metric:
                best_metric = cur_metric
                best_vals = dict_vals
        return best_metric, best_vals


class OptunaOptimizer:
    def __init__(self, args_to_opt, **kwargs):
        self.args_to_opt = args_to_opt
        self.kwargs = kwargs

    def objective(self, trial):
        sampled_args = {}
        for name, (dtype, mn, mx) in self.args_to_opt.items():
            if dtype == float: # suggest it is lr
                sampled_args[name] = trial.suggest_float(name, mn, mx, log=True)
            elif dtype == int:
                sampled_args[name] = trial.suggest_int(name, mn, mx)
            else:
                raise RuntimeError(f"Unexpected dtype {dtype}")
        return run_exp(**self.kwargs, **sampled_args)

    def optimize(self, n_trials: int = 10):
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_trial, study.best_params
