from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.utils import check_random_state, _safe_indexing
from sklearn.base import RegressorMixin, BaseEstimator, ClassifierMixin
import numpy as np
import catboost as cb


def _get_grouped_splits(X, y, sample_weight, validation_fraction, gen_random_seed, groups):
    gss = GroupShuffleSplit(n_splits=2, test_size=validation_fraction, random_state=gen_random_seed)
    train_index, test_index = next(gss.split(X, y, groups=groups))
    X_train = _safe_indexing(X, train_index)
    X_val = _safe_indexing(X, test_index)
    y_train = _safe_indexing(y, train_index)
    y_val = _safe_indexing(y, test_index)

    if sample_weight is None:
        sample_weight_train = sample_weight_val = None
    else:
        sample_weight_train = _safe_indexing(sample_weight, train_index)
        sample_weight_val = _safe_indexing(sample_weight, test_index)

    return (X_train, X_val, y_train, y_val, sample_weight_train, sample_weight_val)
    

def _get_splits_any(X, y, sample_weight, validation_fraction, gen_random_seed):
    if sample_weight is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=validation_fraction,
            random_state=gen_random_seed,
        )
        sample_weight_train = sample_weight_val = None
    else:
        (
            X_train,
            X_val,
            y_train,
            y_val,
            sample_weight_train,
            sample_weight_val,
        ) = train_test_split(
            X,
            y,
            sample_weight,
            test_size=validation_fraction,
            random_state=gen_random_seed,
        )
    return (X_train, X_val, y_train, y_val, sample_weight_train, sample_weight_val)


def _validation_splits(X, y, sample_weight, validation_fraction, gen_random_seed, groups):
    if groups is None:
        return _get_splits_any(X, y, sample_weight, validation_fraction, gen_random_seed)
    else:
        return _get_grouped_splits(X, y, sample_weight, validation_fraction, gen_random_seed, groups)
    


class AESCatBoostRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, base_cat, final_refit=True):
        self.base_cat = base_cat
        self.final_refit = final_refit

    def fit(self, X, y, sample_weight=None, cat_features=None, early_stopping_rounds=10, validation_fraction=0.1, random_state=None, groups=None):
        rng = check_random_state(random_state)
        self._random_seed = rng.randint(np.iinfo(np.uint32).max, dtype="u8")
        (X_train, X_val, y_train, y_val, sample_weight_train, sample_weight_val) = _validation_splits(X, y, sample_weight, validation_fraction, self._random_seed, groups)


        val_pool = cb.Pool(X_val, y_val, weight=sample_weight_val, cat_features=cat_features)
        self.base_cat.fit(X_train, y_train, sample_weight=sample_weight_train, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds, cat_features=cat_features)
        n_estimators = self.base_cat.get_best_iteration()+1
        if self.final_refit:
            base_cat_params = self.base_cat.get_params()
            base_cat_params["n_estimators"] = n_estimators
            base_cat_params["learning_rate"] = self.base_cat.learning_rate_
            self.base_cat = cb.CatBoostRegressor(**base_cat_params)
            self.base_cat.fit(X, y, sample_weight=sample_weight, cat_features=cat_features)

        return self

    def predict(self, X):
        return self.base_cat.predict(X)

class AESCatBoostClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base_cat, final_refit=True):
        self.base_cat = base_cat
        self.final_refit = final_refit

    def fit(self, X, y, sample_weight=None, cat_features=None, early_stopping_rounds=10, validation_fraction=0.1, random_state=None, groups=None):
        rng = check_random_state(random_state)
        self._random_seed = rng.randint(np.iinfo(np.uint32).max, dtype="u8")
        (X_train, X_val, y_train, y_val, sample_weight_train, sample_weight_val) = _validation_splits(X, y, sample_weight, validation_fraction, self._random_seed, groups)


        val_pool = cb.Pool(X_val, y_val, weight=sample_weight_val, cat_features=cat_features)
        self.base_cat.fit(X_train, y_train, sample_weight=sample_weight_train, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds, cat_features=cat_features)
        self.initial_evals = self.base_cat.get_evals_result()
        n_estimators = self.base_cat.get_best_iteration()+1
        
   
        if self.final_refit:
            
            base_cat_params = self.base_cat.get_params()
            base_cat_params["n_estimators"] = n_estimators
            base_cat_params["learning_rate"] = self.base_cat.learning_rate_

            self.base_cat = cb.CatBoostClassifier(**base_cat_params)
            self.base_cat.fit(X, y, sample_weight=sample_weight, cat_features=cat_features)

        return self

    def predict(self, X):
        return self.base_cat.predict(X)
    
    def predict_proba(self, X):
        return self.base_cat.predict_proba(X)



# example:
# base_cat = cb.CatBoostRegressor(random_state=42)
# aes_cb = AESCatBoostRegressor(base_cat)
# aes_cb.fit(X, y, validation_fraction = 0.1, random_state=42, groups=df[group_col])
# aes_cb.predict(X)