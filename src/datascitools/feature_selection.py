from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
from BorutaShap import BorutaShap
from dataclasses import dataclass
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

# make the above a dataclass:
@dataclass
class BorutaShortener:
    n_trials: int = 100
    gpu: bool = False
    n_jobs: int = 8
    classification_fl: bool = False
    sample_fl: bool = True
    base_model: object = None

    def __post_init__(self):
        # validate the base model
        if self.base_model is None:
            if self.gpu:
                self.base_model = CatBoostRegressor(
                    iterations=100, task_type="GPU", devices="0:1"
                )
            else:
                self.base_model = lgb.LGBMRegressor(
                    n_estimators=100, n_jobs=self.n_jobs
                )

        self.feature_selector = BorutaShap(
            model=self.base_model,
            importance_measure="shap",
            classification=self.classification_fl,
        )

    def shorten(
        self,
        df,  # pd.DataFrame
        features,  # list
        label_col,  # str
        sample_weight_col=None,
    ):  # str or None
        # determine cat_features
        cat_features = list(df[features].select_dtypes(include=["category"]).columns)
        # determine text_features
        text_features = list(
            df[features].select_dtypes(include=["string", "object"]).columns
        )
        # determine non_cat_features
        non_cat_features = [
            x for x in features if x not in cat_features and x not in text_features
        ]
        X = df[non_cat_features].apply(lambda x: x.fillna(x.mean()), axis=0)

        if sample_weight_col is None:
            w = None
        else:
            w = df[sample_weight_col]

        self.feature_selector.fit(
            X=X,
            y=df[label_col],
            sample_weight=w,
            n_trials=self.n_trials,
            sample=self.sample_fl,
            train_or_test="test",
            normalize=False,
            verbose=True,
        )

        self.feature_selector.TentativeRoughFix()

        self.boruta_features_ = (
            list(self.feature_selector.Subset().columns) + cat_features + text_features
        )

def shorten_features_catboost(df,
                                features,
                                label_col,
                                n_estimators = 200,
                                sample_weight_col = None,
                                model = None,
                                group_id = None,
                                steps = 6,
                                gpu = False,
                                n_jobs = 8,
                                min_features = 5,
                                exact_feature_count = None,
                                cv_splitter = GroupShuffleSplit(n_splits=2, test_size = 0.4),
                                step_function = 2/3):

    # Determine number of feature sets to try


    if exact_feature_count is not None:
        feature_counts = [exact_feature_count]
    else:
        feature_count = len(features)
        feature_counts = []

        while feature_count > (min_features/step_function):
            feature_count = int(feature_count * step_function)
            feature_counts.append(feature_count)

    print("trying X feature counts:", len(feature_counts))

    if group_id is not None: 
        group_array = df[group_id]
    else: 
        group_array = None

    train_index, test_index = next(cv_splitter.split(df[features], df[label_col], group_array))

    if sample_weight_col is not None:
        w_train, w_test = df.iloc[train_index][sample_weight_col], df.iloc[test_index][sample_weight_col]
    else:
        w_train, w_test = None, None

    cat_features = list(df[features].select_dtypes(include=["category"]).columns)
    text_features = list(df[features].select_dtypes(include=["string","object"]).columns)

    if len(text_features) > 0: print("Text features:", len(text_features))

    train_pool = Pool(df.iloc[train_index][features], label=df.iloc[train_index][label_col], weight = w_train, cat_features = cat_features, text_features = text_features)
    test_pool = Pool(df.iloc[test_index][features], label=df.iloc[test_index][label_col], weight = w_test, cat_features = cat_features, text_features = text_features)

    if model is None:

        if gpu == True:
            model = CatBoostRegressor(iterations=n_estimators, verbose=False, task_type="GPU", devices='0:1', od_type='Iter', od_wait = 20)
        else:
            model = CatBoostRegressor(iterations=n_estimators, verbose=False, thread_count = n_jobs, od_type='Iter', od_wait = 20)

    model_scores = {}
    model_features= {}

    for feature_count in tqdm(feature_counts):
        
        summary = model.select_features(
                                        train_pool,
                                        eval_set=test_pool,
                                        features_for_select=features,     # we will select from all features
                                        num_features_to_select=feature_count,  # we want to select exactly important features
                                        steps=steps,                                     # more steps - more accurate selection
                                        train_final_model=False,                          # to train model with selected features
                                        logging_level='Silent',
                                        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
                                        shap_calc_type=EShapCalcType.Regular,
                                        plot=False)

        short_features = summary['selected_features_names']
        short_cat_features = list(set(short_features) & set(cat_features))
        
        train_pool_short = Pool(df.iloc[train_index][short_features], label=df.iloc[train_index][label_col], weight = w_train, cat_features = short_cat_features)
        test_pool_short = Pool(df.iloc[test_index][short_features], label=df.iloc[test_index][label_col], weight = w_test, cat_features = short_cat_features)
        model.fit(train_pool_short, eval_set=test_pool_short, early_stopping_rounds=10, verbose = False)
        score = list(model.best_score_['validation'].values())[0]

        model_scores[feature_count] = score
        model_features[feature_count] = short_features

    min_key = min(model_scores, key=model_scores.get)
    best_features = model_features[min_key]

    return best_features