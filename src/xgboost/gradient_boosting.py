import time
import pickle

import cupy
import numpy as np
import optuna
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier

import data

mempool = cupy.get_default_memory_pool()

with cupy.cuda.Device(0):
    mempool.set_limit(size=1024**3)


class TreeWrapper(XGBClassifier):
    def __init__(self, gamma, min_child_weight, n_estimators, subsample, learning_rate, early_stopping_rounds):
        super().__init__(
            gamma=gamma,
            subsample=subsample,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            colsample_bytree=0.919,
            max_depth=10,
            max_leaves=0,
            booster='gbtree',
            device="cuda",
            tree_method='hist',
            sampling_method='gradient_based',
            objective='multi:softmax',
            grow_policy='lossguide',
            early_stopping_rounds=early_stopping_rounds,
            random_state=42
        )


def optimize(trial: optuna.Trial):
    print("\n[+] Init...")
    gamma = trial.suggest_float('gamma', low=0, high=1)
    subsample = trial.suggest_float('subsample', low=0.6, high=1)
    learning_rate = trial.suggest_float(
        'learning_rate', low=0.01, high=0.5)
    n_estimators = trial.suggest_int(
        'n_estimators', low=100, high=2000)
    min_child_weight = trial.suggest_float(
        'min_child_weight', low=1, high=7)
    scores = []
    for fold_indx in range(10):
        X_train, y_train, X_test, y_test, sample_weight = data.gen_fold_training(
            fold_indx)
        print("[!] Training...")
        start = time.monotonic()
        tree = TreeWrapper(gamma, min_child_weight,
                           n_estimators, subsample, learning_rate, 10)
        tree.fit(X_train, y_train, sample_weight=sample_weight,
                 eval_set=[(X_test, y_test)], verbose=0)
        end = time.monotonic()
        print(f"[!] Training takes {end - start} sec.")
        print("[+] Evaluating...")
        score = f1_score(y_test.get(), tree.predict(X_test), average='macro')
        trial.report(score, fold_indx)
        if trial.should_prune():
            raise optuna.TrialPruned()
        scores.append(score)
    print()
    return np.mean(scores)


if __name__ == '__main__':
    study = optuna.create_study(
        study_name='xgboost',
        direction="maximize",
        storage='sqlite:///optuna_study.db',
        load_if_exists=True
    )
    study.optimize(optimize, n_trials=100, show_progress_bar=True)
    print("\nBest params:")
    print(f"gamma = {study.best_params['gamma']}")
    print(f"min_child_weight = {study.best_params['min_child_weight']}")
    print(f"n_estimators = {study.best_params['n_estimators']}")
    print(f"subsample = {study.best_params['subsample']}")
    print(f"learning_rate = {study.best_params['learning_rate']}")
    # Validate the best model:
    X_train, y_train, X_val, y_val = data.load_dataset()
    tree = TreeWrapper(
        study.best_params['gamma'],
        study.best_params['min_child_weight'],
        study.best_params['n_estimators'],
        study.best_params['subsample'],
        study.best_params['learning_rate'],
        0
    )
    tree.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    y_pred = tree.predict(X_val)
    print(classification_report(y_val.get(), y_pred))
    tree.save_model('gradient_boost_tree.json')
    params = study.best_params
    with open('best_params_tree.pickle', 'wb') as f:
        pickle.dump(params, f)
