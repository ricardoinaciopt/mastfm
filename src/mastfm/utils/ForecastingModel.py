import re
import pandas as pd
from mlforecast.utils import PredictionIntervals
from mlforecast.target_transforms import Differences
from mlforecast import MLForecast
from copy import deepcopy


class ForecastingModel:
    """
    Class for training and forecasting using a specified model or list of models with hyperparameter tuning.
    It takes sklearn-compatible regressors to make forecasts. An interval of confidence can be specified.
    """

    def __init__(
        self,
        model,
        train_set,
        frequency,
        seasonality,
        horizon,
        level,
        target_differences,
    ):
        """
        Initializes the ForecastingModel object.
        """
        self.model = deepcopy(model)
        self.model_name = self.__simplify_name(model.__class__.__name__)
        self.train_set = train_set
        self.horizon = horizon
        self.frequency = frequency
        self.seasonality = seasonality
        self.level = level
        self.target_differences = target_differences
        self.regressor = None
        self.predictions = None
        self.categorical_models = [
            "lgbm",
            "xgb",
        ]
        self.non_categorical_models = [
            "ridge",
            "lasso",
            "linearregression",
            "elasticnet",
            "randomforest",
        ]

        self.fit()

    def fit(self):
        """
        Fits the forecasting model.
        """

        fit_kwargs = {
            "df": self.train_set,
            "prediction_intervals": PredictionIntervals(h=self.horizon),
        }

        # handle categorical features
        if self.model_name in self.non_categorical_models:
            fit_kwargs["static_features"] = ["unique_id"]
        if "enable_categorical" in self.model.__dict__:
            self.model.set_params(enable_categoric=True)

        target_transforms = None
        if self.target_differences is not None:
            target_transforms = [(Differences([self.target_differences]))]

        self.regressor = MLForecast(
            models=self.model,
            freq=self.frequency,
            lags=[self.horizon],
            target_transforms=target_transforms,
        ).fit(**fit_kwargs)
        self.forecast()

    def forecast(self):
        """
        Generates forecasts using the fitted model.
        """
        if not self.level:
            raise ValueError("'level' must be defined to compute prediction intervals.")
        self.predictions = self.__simplify_name(
            self.regressor.predict(h=self.horizon, level=[self.level])
        )

    def __simplify_name(self, data):
        """
        Simplifies the name of the employed model, for improved readability.
        """
        if isinstance(data, pd.DataFrame):
            data.columns = [
                re.compile(r"\([^)]*\)")
                .sub("", str(col))
                .lower()
                .replace("regressor", "")
                .replace("auto", "")
                .strip()
                for col in data.columns
            ]
        else:
            data = (
                re.compile(r"\([^)]*\)")
                .sub("", data)
                .lower()
                .replace("regressor", "")
                .replace("auto", "")
                .strip()
            )
        return data
