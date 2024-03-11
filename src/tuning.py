import optuna
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def _objective_xgb(trial, x_train, y_train, x_val, y_val):
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.3, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "eval_metric": "auc",
    }
    
    xgb_clf = XGBClassifier(**params, random_state=42).fit(x_train, y_train, verbose=False)
    y_pred = xgb_clf.predict_proba(x_val)
    roc_auc = roc_auc_score(y_val, y_pred[:,1])
    return roc_auc

def _objective_cat(trial, x_train, y_train, x_val, y_val):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1),
        "depth": trial.suggest_int("max_depth", 2, 8),
        "iterations": trial.suggest_int("n_estimators", 50, 150),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 1),
        }

    cat_clf = CatBoostClassifier(**params, random_state=42).fit(x_train, y_train, verbose=False)
    y_pred = cat_clf.predict_proba(x_val)
    roc_auc = roc_auc_score(y_val, y_pred[:,1])
    return roc_auc

def _objective_random_forest(trial, x_train, y_train, x_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }

    rf_clf = RandomForestClassifier(**params, random_state=42).fit(x_train, y_train)
    y_pred = rf_clf.predict_proba(x_val)
    roc_auc = roc_auc_score(y_val, y_pred[:,1])
    return roc_auc

def _objective_lr(trial, x_train, y_train, x_val, y_val):
    params = {
        "solver": trial.suggest_categorical("solver", ["lbfgs", "sag", "saga", "liblinear", "newton-cg"]),
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "C": trial.suggest_float("C", 0.01, 10.0),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2),
    }

    rf_clf = RandomForestClassifier(**params, random_state=42).fit(x_train, y_train)
    y_pred = rf_clf.predict_proba(x_val)
    roc_auc = roc_auc_score(y_val, y_pred[:,1])
    return roc_auc

def _objective_mlp(trial, x_train, y_train, x_val, y_val):
    params = {
        "hidden_layer_sizes": trial.suggest_int("hidden_layer_sizes", 10, 200),
        "activation": trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"]),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1),
        "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
        "early_stopping": trial.suggest_categorical("early_stopping", [True, False]),
        "validation_fraction": trial.suggest_float("validation_fraction", 0.1, 0.3),
    }

    rf_clf = RandomForestClassifier(**params, random_state=42).fit(x_train, y_train)
    y_pred = rf_clf.predict_proba(x_val)
    roc_auc = roc_auc_score(y_val, y_pred[:,1])
    return roc_auc

def find_best_model(model, x_train, y_train, x_val, y_val, trials=100):

    
    study = optuna.create_study(direction="maximize")

    if model == "xgb":
        study.optimize(lambda trial: _objective_xgb(trial, x_train, y_train, x_val, y_val), n_trials=trials, show_progress_bar=True)
    
    elif model == "cat":
        study.optimize(lambda trial: _objective_cat(trial, x_train, y_train, x_val, y_val), n_trials=trials, show_progress_bar=True)
    
    elif model == "random_forest":
        study.optimize(lambda trial: _objective_random_forest(trial, x_train, y_train, x_val, y_val), n_trials=trials, show_progress_bar=True)

    elif model == "lr":
        study.optimize(lambda trial: _objective_lr(trial, x_train, y_train, x_val, y_val), n_trials=trials, show_progress_bar=True)
    
    elif model == "mlp":
        study.optimize(lambda trial: _objective_mlp(trial, x_train, y_train, x_val, y_val), n_trials=trials, show_progress_bar=True)
    
    else:
        raise("Model type not found")

    
    best_model = study.best_trial
    return best_model