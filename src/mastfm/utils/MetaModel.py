import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

from imblearn.over_sampling import (
    SMOTE,
    ADASYN,
    SVMSMOTE,
    BorderlineSMOTE,
)


class MetaModel:
    """
    MetaModel class for training a metamodel using LightGBM or XGBoost with optional data resampling.
    """

    def __init__(
        self,
        train_set,
        columns_to_drop,
        resampler=None,
        target=None,
        tuning=True,
    ):
        """
        Initializes the MetaModel object.
        """
        self.resampler = resampler
        self.tuning = tuning
        self.train = train_set
        self.columns_to_drop = columns_to_drop
        self.X = None
        self.y = None
        self.resamplers = {
            "SMOTE": SMOTE(),
            "ADASYN": ADASYN(),
            "SVMSMOTE": SVMSMOTE(),
            "BorderlineSMOTE": BorderlineSMOTE(),
        }
        self.classifier = None
        self.target = target
        self.preprocess_set()
        self.fit_model()

    def preprocess_set(self):
        """
        Preprocesses the training set, including data resampling if specified.
        """
        if "unique_id" in self.train.columns:
            self.train.set_index("unique_id", inplace=True)
        if "metric" in self.train.columns:
            self.train.drop(
                columns=["metric"], inplace=True
            )  # remove this categorical feature first to allow resampling
        self.train.fillna(0, inplace=True)

        if self.resampler and self.resampler in self.resamplers:
            data_to_resample = self.train.copy()
            threshold_mask = self.train[self.target].astype(int)

            resampled_data, err_class = self.resamplers[self.resampler].fit_resample(
                data_to_resample, threshold_mask
            )

            self.X = resampled_data.drop(
                [col for col in self.columns_to_drop if col in resampled_data.columns],
                axis=1,
            )
            self.y = resampled_data[self.target]
        else:
            self.X = self.train.drop(
                [col for col in self.columns_to_drop if col in self.train.columns],
                axis=1,
            )
            self.y = self.train[self.target]

    def fit_model(self, model="LGBM"):
        """
        Fits the classifier model.
        """
        lgbm_params = {"n_estimators": 200, "verbosity": -1}
        xgb_params = {"n_estimators": 200, "verbosity": 0}

        if model == "LGBM":
            param_grid = {
                "num_leaves": [3, 5, 10, 15],
                "max_depth": [-1, 3, 5, 10, 15],
                "lambda_l1": [0.1, 1, 10, 100],
                "lambda_l2": [0.1, 1, 10, 100],
                "learning_rate": [0.05, 0.1, 0.2],
                "min_child_samples": [7, 15, 30],
            }
            estimator = lgb.LGBMClassifier(**lgbm_params)
        elif model == "XGB":
            param_grid = {
                "max_depth": [3, 5, 10, 15],
                "lambda": [0.1, 1, 10, 100],
                "alpha": [0.1, 1, 10, 100],
                "learning_rate": [0.05, 0.1, 0.2],
                "min_child_weight": [7, 15, 30],
            }
            estimator = xgb.XGBClassifier(**xgb_params)
        else:
            print("Unsupported algorithm, use: LGBM or XGB.")
            return

        if self.tuning:
            rnd_search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                scoring="roc_auc",
                n_jobs=-1,
            )
            rnd_search.fit(self.X, self.y)

            self.classifier = rnd_search.best_estimator_
        else:
            self.classifier = estimator.fit(self.X, self.y)
