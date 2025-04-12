import os
import shap
import shapiq
import inspect
import __main__
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
from tsfeatures import tsfeatures
from utilsforecast.losses import *
from utilsforecast.evaluation import evaluate
from utilsforecast.plotting import plot_series
from sklearn.preprocessing import StandardScaler


from .utils.MetaModel import MetaModel
from .utils.PrepareDataset import PrepareDataset
from .utils.ForecastingModel import ForecastingModel
from .utils.TimeSeriesGenerator import TimeSeriesGenerator


class MASTFM:
    """
    Meta-learning and data Augmentation for Stress Testing Forecasting Models(MASTFM).
    Class for model evaluation and feature extraction.
    Computes which timeseries put the forecasting model under stress.
    Identifies instances (timeseries) of large errors and large uncertainty.
    """

    def __init__(
        self,
        forecasting_model,
        horizon,
        frequency,
        seasonality,
        quantile=None,
        level=None,
        metric=None,
        augmentation_method=None,
        target=None,
    ):
        if self._is_valid_context():
            self.__initialize_attributes(
                forecasting_model,
                horizon,
                frequency,
                seasonality,
                quantile,
                level,
                metric,
                augmentation_method,
                target,
            )
        else:
            raise ValueError(
                "Please run MASTFM from a Jupyter Notebook or from the '__main__' method of a Python script."
            )

    def _is_valid_context(self):
        is_notebook = not hasattr(__main__, "__file__")
        in_main_function = self._is_in_main_function()
        return is_notebook or in_main_function

    def _is_in_main_function(self):
        frame = inspect.currentframe()
        while frame:
            if frame.f_code.co_name == "main":
                return True
            frame = frame.f_back
        return False

    def __initialize_attributes(
        self,
        forecasting_model,
        horizon,
        frequency,
        seasonality,
        quantile,
        level,
        metric,
        augmentation_method,
        target,
    ):
        """
        Initializes the attributes of MASTFM.
        """
        self.forecasting_model = forecasting_model
        self.horizon = horizon
        self.frequency = frequency
        self.seasonality = seasonality
        self.quantile = quantile / 100 or 0.8
        self.level = level or quantile * 100
        self.__metric_name = metric or "smape"
        self.__metric = self.__initialize_metrics(self.__metric_name)
        if target is not None and target not in [
            "errors",
            "uncertainty",
            "hubris",
        ]:
            raise ValueError(
                f"Invalid target: {target}. Please choose one of: 'errors', 'uncertainty', or 'hubris'."
            )
        self.__target_name = target
        self.target = {
            "errors": "large_error",
            "uncertainty": "large_uncertainty",
            "hubris": "le_lc",
        }.get(target, "large_error")
        self.error_summary = None
        self.metamodel = None
        self.__merged_forecasts = None
        self.__large_errors_df = None
        self.large_errors_ids = None
        self.__interval_summary = None
        self.__avg_uncertainty_df = None
        self.__large_uncertainty_df = None
        self.large_uncertainty_ids = None
        self.__features_errors = None
        self.__features_stress = None
        self.__train_features = None
        self.__feature_names = None
        self.__features_index = None
        self.__model_predictions = None
        self.__large_certainty_df = None
        self.hubris_ids = None
        self.hubris_df = None
        self.large_certainty_ids = None
        self.augmentation_method = augmentation_method or "ADASYN"
        self.oversamplers = ["SMOTE", "ADASYN", "BorderlineSMOTE", "SVMSMOTE"]
        self.generators = [
            "TSMixup",
            "DBA",
            "Jittering",
            "SeasonalMBB",
            "Scaling",
            "MagnitudeWarping",
            "TimeWarping",
        ]
        self.__model_name = None
        self.cols_to_drop = [
            "large_error",
            "large_uncertainty",
            "large_certainty",
            "le_lc",
            "class",
        ]
        self.__shap_values = None

    def fit(self, df, val_size=None, target_differences=None):
        if df is None:
            raise ValueError("Data for fitting must be specified.")
        if val_size is None:
            val_size = self.horizon
        with tqdm(
            total=6,
            desc="Initializing MASTFM",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            leave=False,
        ) as pbar:
            pbar.update(1)
            pbar.set_description("Preparing data")
            self.__prepare_datasets(df=df, val_size=val_size)
            pbar.update(1)
            pbar.set_description("Carrying out performance estimation")
            self.__initialize_model_predictions(self.dev_set, target_differences)
            pbar.update(1)
            pbar.set_description("Extracting features")
            self.__evaluate_and_extract_features(X=self.dev_set, y=self.valid_set)
            pbar.update(1)
            pbar.set_description("Computing uncertainty and certainty")
            self.__compute_uncertainty_and_certainty(self.train_set)
            pbar.update(1)
            pbar.set_description("Fitting metamodel")
            self.__fit_metamodel()
            pbar.update(1)
            pbar.set_description("Applying metamodel")
            self.__apply_metamodel()

    def __prepare_datasets(self, df, val_size):
        """
        Prepares the train, development, and validation datasets.
        """
        dataset = PrepareDataset(data=df)
        dataset.split_train_set(val_size=val_size)

        self.train_set = dataset.train
        self.dev_set = dataset.dev_set
        self.valid_set = dataset.valid

    def __initialize_model_predictions(self, train_set=None, target_differences=None):
        """
        Initializes model predictions using the forecasting model.
        """
        fm = ForecastingModel(
            model=self.forecasting_model,
            train_set=train_set,
            frequency=self.frequency,
            seasonality=self.seasonality,
            horizon=self.horizon,
            level=self.level,
            target_differences=target_differences,
        )
        self.__model_predictions = [fm.predictions]
        self.__model_name = fm.model_name

    def __evaluate_and_extract_features(self, X, y=None, is_dev=True):
        """
        Evaluates forecasts and extracts features from the dataset.
        """
        if is_dev:
            self.__evaluate_forecasts(X=X, y=y, is_dev=is_dev)
            self.__get_large_errors(
                model=self.__model_name,
                metric=self.__metric_name,
                quantile=self.quantile,
            )
        self.__extract_features(train_set=X, is_dev=is_dev)

    def __compute_uncertainty_and_certainty(self, train_set=None):
        """
        Computes uncertainty, and identifies stress-inducing instaces of it, alongside certainty, for the given train set.
        """
        self.__compute_uncertainty(
            train_set=train_set, predictions=self.__merged_forecasts, level=self.level
        )
        self.__get_large_uncertainty(model=self.__model_name, quantile=self.quantile)
        self.__get_large_certainty(
            model=self.__model_name, quantile=(1 - self.quantile)
        )

    def __fit_metamodel(self):
        """
        Initializes the metamodel based on the augmentation method.
        Splits the metamodel train set 70/30, leaving the latter out for calibration.
        """
        self.__build_stress_set()
        self.cols_to_drop += [f"avg_interval_size_{self.__model_name}"]
        mm_full = self.__features_stress.copy()
        mm_full.set_index("unique_id", inplace=True)
        split_index = int(len(mm_full) * 0.7)
        mm_train = mm_full.iloc[:split_index]
        mm_cal = mm_full.iloc[split_index:]
        if self.augmentation_method in self.oversamplers:
            self.metamodel = MetaModel(
                train_set=mm_train,
                resampler=self.augmentation_method,
                columns_to_drop=self.cols_to_drop,
                target=self.target,
                quantile=self.quantile,
            )
        elif self.augmentation_method in self.generators:
            self.__initialize_generator_metamodel(mm_train, mm_cal)
        else:
            raise ValueError(
                f"Unknown augmentation method: {self.augmentation_method}. Please select a valid oversampling method: 'SMOTE', 'ADASYN', 'BorderlineSMOTE', 'SVMSMOTE', or a valid generation method:  'TSMixup', 'DBA', 'Jittering', 'SeasonalMBB', 'Scaling', 'MagnitudeWarping', 'TimeWarping'."
            )

    def __build_stress_set(self):
        """
        Builds the stress set by combining features and stress indicators.
        """
        self.__features_stress = self.__features_errors
        self.__features_stress[f"avg_interval_size_{self.__model_name}"] = (
            self.__avg_uncertainty_df[f"avg_interval_size_{self.__model_name}"]
        )
        self.__features_stress["large_uncertainty"] = self.__features_stress[
            "unique_id"
        ].apply(lambda x: 1 if x in self.large_uncertainty_ids else 0)
        self.__features_stress["large_certainty"] = self.__features_stress[
            "unique_id"
        ].apply(lambda x: 1 if x in self.large_certainty_ids else 0)
        self.__features_stress["le_lc"] = (
            (self.__features_stress["large_error"] == 1)
            & (self.__features_stress["large_certainty"] == 1)
        ).astype(int)

    def __initialize_generator_metamodel(self, mm_train, mm_cal):
        """
        Initializes the generator metamodel using semisynthetic time series data.
        """
        gen_set = self.dev_set
        # remove series present in the calibration set
        gen_set = gen_set[~gen_set["unique_id"].isin(mm_cal.index)]
        mm_train_stress_ids = self.__get_train_stress_ids(mm_train=mm_train)
        gen_set[self.target] = gen_set["unique_id"].apply(
            lambda x: 1 if x in mm_train_stress_ids else 0
        )
        gen_set["unique_id"] = gen_set["unique_id"].cat.remove_unused_categories()
        min_len = gen_set["unique_id"].value_counts().min()
        max_len = gen_set["unique_id"].value_counts().max()
        augmenter = TimeSeriesGenerator(
            gen_set,
            seasonality=self.seasonality,
            frequency=self.frequency,
            min_len=min_len,
            max_len=max_len,
            target=self.target,
            quantile=self.quantile,
        )
        synthetic_ts_df = augmenter.generate_synthetic_dataset(
            method_name=self.augmentation_method
        )
        synthetic_ts_df[self.target] = 1
        gen_df = pd.concat(
            [gen_set.reset_index(drop=True), synthetic_ts_df.reset_index(drop=True)],
            ignore_index=True,
        )
        gen_df = gen_df.drop(columns=self.target)
        features = tsfeatures(gen_df, freq=self.seasonality)
        features[self.target] = features["unique_id"].apply(
            lambda x: 1 if x in mm_train_stress_ids or "SYN" in x else 0
        )
        features = self.__balance_stress(features)
        features = features.fillna(0)
        self.metamodel = MetaModel(
            train_set=features,
            columns_to_drop=self.cols_to_drop,
            target=self.target,
        )

    def __get_train_stress_ids(self, mm_train):
        """
        Returns the set of ids form stress-inducing time series, according to a specific target.
        """
        target_map = {
            "large_error": self.large_errors_ids,
            "large_uncertainty": self.large_uncertainty_ids,
            "le_lc": self.__features_stress["unique_id"][
                self.__features_stress["le_lc"] == 1
            ].tolist(),
        }
        try:
            target_ids = target_map[self.target]
        except KeyError:
            raise ValueError("Invalid target type")

        mm_train_stress_ids = [id for id in target_ids if id in mm_train.index]
        return mm_train_stress_ids

    def __initialize_metrics(self, metric):
        """
        Defines the metric to compute the prediction errors.
        """
        metric_dict = {
            "mase": partial(mase, seasonality=self.seasonality),
            "smape": smape,
        }
        return [metric_dict[metric]]

    def __merge_predictions(self, y, is_dev):
        """
        Merges model predictions with the test set.
        """
        self.__merged_forecasts = y
        if is_dev:
            self.__merge_with_predictions(self.__model_predictions, "inner")
        else:
            self.__initialize_model_predictions(self.train_set)
            self.__merge_with_predictions(self.__model_predictions, "inner")

    def __merge_with_predictions(self, predictions_list, how):
        """
        Helper function to merge the forecasts with predictions.
        """
        for prediction_df in predictions_list:
            self.__merged_forecasts = pd.merge(
                self.__merged_forecasts,
                prediction_df,
                on=["unique_id", "ds"],
                how=how,
            )

    def __evaluate_forecasts(self, X, y, is_dev):
        """
        Evaluates forecasts using predefined metrics.
        """
        self.__merge_predictions(y=y, is_dev=is_dev)
        self.evaluation = evaluate(
            self.__merged_forecasts,
            metrics=self.__metric,
            train_df=X,
        )
        self.error_summary = (
            self.evaluation.drop(columns="unique_id")
            .groupby("metric")
            .mean()
            .reset_index()
        )
        self.cols_to_drop += self.error_summary.columns.to_list()

    def __get_large_errors(self, model, metric, quantile=None):
        """
        Identifies large errors in the predictions.
        """
        self.errors_df = self.evaluation
        self.errors_df = self.errors_df[self.errors_df["metric"] == metric]
        if quantile is not None:
            percentile_n = self.errors_df[model].quantile(quantile)
            self.__large_errors_df = self.errors_df[
                (self.errors_df[model] > percentile_n)
            ]
        self.large_errors_ids = self.__large_errors_df["unique_id"].unique()

    def __compute_uncertainty(self, train_set, level, predictions=None):
        """
        Computes the average of the uncertainty intervals for predictions.
        """
        if predictions is None or predictions.empty:
            if not self.__merged_forecasts.empty:
                predictions = self.__merged_forecasts
            else:
                raise ValueError(
                    'No forecasts available, please use the "evaluate_forecasts()" method first.'
                )
        model_columns = predictions.columns[3:-1]
        models = set(col.split("-")[0] for col in model_columns if "-" in col)
        avg_uncertainty_df = pd.DataFrame()

        for model in models:
            concat_list = []
            lo_col = f"{model}-lo-{level}"
            hi_col = f"{model}-hi-{level}"
            lo_col_scaled = f"{model}-lo-{level}-scaled"
            hi_col_scaled = f"{model}-hi-{level}-scaled"
            for id in predictions["unique_id"].unique():
                scaler = StandardScaler()
                scaler.fit(train_set[train_set["unique_id"] == id][["y"]].values)
                scaled_intervals = predictions[predictions["unique_id"] == id]
                scaled_intervals[lo_col_scaled] = scaler.transform(
                    scaled_intervals[[lo_col]]
                )
                scaled_intervals[hi_col_scaled] = scaler.transform(
                    scaled_intervals[[hi_col]]
                )
                scaled_intervals["scaled_interval_size"] = (
                    scaled_intervals[hi_col_scaled] - scaled_intervals[lo_col_scaled]
                )
                concat_list.append(
                    scaled_intervals[
                        [
                            "unique_id",
                            "ds",
                            lo_col_scaled,
                            hi_col_scaled,
                            "scaled_interval_size",
                        ]
                    ]
                )
            concat_df = pd.concat(concat_list)
            model_avg_uncertainty = (
                concat_df.groupby("unique_id")["scaled_interval_size"]
                .mean()
                .rename(f"avg_interval_size_{model}")
            )
            avg_uncertainty_df = pd.concat(
                [avg_uncertainty_df, model_avg_uncertainty], axis=1
            )

        self.__avg_uncertainty_df = avg_uncertainty_df.reset_index()
        self.__avg_uncertainty_df.rename(columns={"index": "unique_id"}, inplace=True)
        self.__interval_summary = pd.DataFrame()
        for col in self.__avg_uncertainty_df.columns[1:]:
            avg_size = self.__avg_uncertainty_df[col].mean()
            model_name = col.replace("avg_interval_size_", "interval_summary_")
            self.__interval_summary[model_name] = [round(avg_size, 3)]
        return self.__avg_uncertainty_df

    def __get_large_uncertainty(self, model, quantile):
        """
        Identifies predictions with large uncertainty intervals for a specified model.
        """
        col = f"avg_interval_size_{model}"
        if col not in self.__avg_uncertainty_df.columns:
            raise ValueError(f"{model} not found.")

        if quantile is not None:
            percentile_n_i = self.__avg_uncertainty_df[col].quantile(quantile)
            self.__large_uncertainty_df = self.__avg_uncertainty_df[
                (self.__avg_uncertainty_df[col] > percentile_n_i)
            ]
            self.large_uncertainty_ids = self.__large_uncertainty_df[
                "unique_id"
            ].unique()
            return self.large_uncertainty_ids
        raise ValueError("'quantile' must be defined.")

    def __get_large_certainty(self, model, quantile):
        """
        Identifies predictions with large certainty intervals for a specified model.
        """
        col = f"avg_interval_size_{model}"
        if col not in self.__avg_uncertainty_df.columns:
            raise ValueValueError(f"{model} not found.")

        if quantile is not None:
            percentile_n_i = self.__avg_uncertainty_df[col].quantile(quantile)
            self.__large_certainty_df = self.__avg_uncertainty_df[
                (self.__avg_uncertainty_df[col] <= percentile_n_i)
            ]
            self.large_certainty_ids = self.__large_certainty_df["unique_id"].unique()

            # also define hubris as the intersection of large errors and large certainty
            self.hubris_ids = list(
                set(self.large_errors_ids).intersection(set(self.large_certainty_ids))
            )
            self.__hubris_df = pd.merge(
                self.__large_errors_df, self.__large_certainty_df, on="unique_id"
            )

            return self.large_certainty_ids
        raise ValueError("'quantile' must be defined.")

    def __extract_features(self, train_set, is_dev=False):
        """
        Extracts features from the training set and merges them with error data.
        """
        features = tsfeatures(train_set, freq=self.seasonality)
        if is_dev:
            self.__features_errors = pd.merge(
                self.errors_df,
                features,
                on="unique_id",
                how="inner",
            )
            self.__features_errors.fillna(0, inplace=True)
            self.__features_errors["large_error"] = (
                self.__features_errors["unique_id"]
                .isin(self.large_errors_ids)
                .astype(int)
            )
        else:
            self.__train_features = features
            self.__train_features.fillna(0, inplace=True)
            self.__train_features.set_index("unique_id", inplace=True)
            self.__feature_names = self.__train_features.columns.to_list()
            self.__features_index = self.__train_features.index.to_list()

    def __balance_stress(self, df, prefix="SYN"):
        """
        Balances the dataset by the target variable.
        """
        df_0 = df[df[self.target] == 0]
        df_1 = df[df[self.target] == 1]

        df_1_with_prefix = df_1[df_1["unique_id"].str.contains(prefix, na=False)]
        df_1_without_prefix = df_1[~df_1["unique_id"].str.contains(prefix, na=False)]
        if len(df_1_with_prefix) < len(df_0):
            raise ValueError(
                "Not enough synthetic samples to balance the dataset. "
                f"Please relax the stress threshold ('quantile' < {int(self.quantile*100)}), or change the 'augmentation_method'."
            )
        df_1_sampled = df_1_with_prefix.sample(
            n=(len(df_0) - len(df_1_without_prefix)), random_state=42
        )
        balanced_df_1 = pd.concat([df_1_without_prefix, df_1_sampled])
        balanced_df = (
            pd.concat([df_0, balanced_df_1])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

        return balanced_df

    def __apply_metamodel(self):
        """
        Applies the metamodel on the predictions.
        """
        self.__evaluate_and_extract_features(X=self.train_set, is_dev=False)
        explainer = shap.Explainer(self.metamodel.classifier)
        self.__shap_values = explainer(self.__train_features)

    def visual_explanations(self, top_n=10, show=True, save=False, relationships=False):
        """
        Produces visual explanations using SHAP values for the top N features.
        If shapiq is True, uses the shapiq package for interaction values.
        """
        if self.__shap_values is None:
            raise ValueError("SHAP values not available")

        if save and not os.path.exists("figures"):
            os.makedirs("figures")

        if relationships:

            explainer = shapiq.TreeExplainer(
                model=self.metamodel.classifier, index="k-SII", min_order=1, max_order=3
            )
            interaction_values = explainer.explain(self.__train_features.iloc[100])

            plt.figure(figsize=(10, 16))
            shapiq.network_plot(
                first_order_values=interaction_values.get_n_order_values(1),
                second_order_values=interaction_values.get_n_order_values(2),
                feature_names=self.__train_features.columns,
            )
            plt.tight_layout()
            if save:
                plt.savefig(
                    "figures/shapiq_network_plot.pdf", bbox_inches="tight", dpi=500
                )
            if show:
                plt.show()
            plt.close()
        else:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                self.__shap_values,
                self.__train_features,
                plot_type="bar",
                max_display=top_n,
                show=False,
            )
            plt.tight_layout()
            if save:
                plt.savefig(
                    "figures/shap_summary_bar.pdf", bbox_inches="tight", dpi=500
                )
            if show:
                plt.show()
            plt.close()

            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                self.__shap_values, self.__train_features, max_display=top_n, show=False
            )
            plt.tight_layout()
            if save:
                plt.savefig("figures/shap_summary.pdf", bbox_inches="tight", dpi=500)
            if show:
                plt.show()
            plt.close()

    def textual_explanations(self, unique_id=None):
        if self.__shap_values is None:
            raise ValueError("SHAP values not available")
        if unique_id is None:
            sample_shap_values = self.__shap_values.values.mean(axis=0)
            sample_base_value = self.__shap_values.base_values.mean()
            text = f"When predicting the outcome on average across all time series, the base value is {sample_base_value:.2f}, which is the average model output when no feature is considered. Lets assess how each feature contributes to shift the model from the base value: \n"
        else:
            if unique_id not in self.__features_index:
                raise ValueError(
                    f"Invalid identifier: {unique_id}. Please provide a valid 'unique_id'."
                )
            sample_index = self.__features_index.index(unique_id)
            sample_shap_values = self.__shap_values.values[sample_index]
            sample_base_value = self.__shap_values.base_values[sample_index]
            text = f'When predicting the outcome for time series "{unique_id}", the base value is {sample_base_value:.2f}, which is the average model output when no feature is considered. Lets assess how each feature contributes to shift the model from the base value: \n'
        feature_contributions = []
        for feature_index, feature_name in enumerate(self.__feature_names):
            shap_value = round(sample_shap_values[feature_index], 2)
            feature_contributions.append((feature_name, shap_value))

        feature_contributions.sort(key=lambda x: x[1], reverse=True)

        for feature_name, shap_value in feature_contributions:
            if shap_value > 0:
                text += f"\tThe feature '{feature_name}' contributes positively by {shap_value:.2f}, indicating that it contributes to stress. \n"
            elif shap_value < 0:
                text += f"\tThe feature '{feature_name}' contributes negatively by {abs(shap_value):.2f}, indicating that it inhibits stress. \n"
            else:
                text += f"\tThe feature '{feature_name}' has a negligible impact on stress. \n"
        print(text)

    def show_large_errors_ids(self, df=False):
        """
        Displays the unique IDs of the timeseries with large errors.
        """
        print(list(self.large_errors_ids))
        if df:
            return self.__large_errors_df

    def show_large_certainty_ids(self, df=False):
        """
        Displays the unique IDs of the timeseries with large certainty.
        """
        print(list(self.large_certainty_ids))
        if df:
            return self.__large_certainty_df

    def show_large_uncertainty_ids(self, df=False):
        """
        Displays the unique IDs of the timeseries with large uncertainty.
        """
        print(list(self.large_uncertainty_ids))
        if df:
            return self.__large_uncertainty_df

    def show_hubris_ids(self, df=False):
        """
        Displays the unique IDs of the timeseries with large errors and large certainty.
        """
        print(list(self.hubris_ids))
        if df:
            return self.__hubris_df

    def plot_stress(self, show=True, save=False):
        """
        The stress classes are determined based on the combination of large error
        and large uncertainty flags in the `self.__features_stress` DataFrame. The
        classes are:
            - 0: No Stress
            - 1: Large Error
            - 2: Large Uncertainty
            - 3: Large Error & Large Uncertainty
            - 4: Hubris
        The method scales the error and uncertainty values to a unit range before plotting.
        """
        conditions = [
            (self.__features_stress["large_error"] == 1)
            & (self.__features_stress["large_certainty"] == 1),
            (self.__features_stress["large_error"] == 0)
            & (self.__features_stress["large_uncertainty"] == 0),
            (self.__features_stress["large_error"] == 1)
            & (self.__features_stress["large_uncertainty"] == 0),
            (self.__features_stress["large_error"] == 0)
            & (self.__features_stress["large_uncertainty"] == 1),
            (self.__features_stress["large_error"] == 1)
            & (self.__features_stress["large_uncertainty"] == 1),
        ]
        choices = [4, 0, 1, 2, 3]

        self.__features_stress["class"] = np.select(conditions, choices)

        x_error_scaled = self.__scale_to_unit_range(
            self.__features_stress[self.__model_name]
        )
        y_uncertainty_scaled = self.__scale_to_unit_range(
            self.__features_stress[f"avg_interval_size_{self.__model_name}"]
        )

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        class_labels = [
            "No Stress",
            "Large Error",
            "Large Uncertainty",
            "Large Error & Large Uncertainty",
            "Hubris",
        ]
        shapes = ["o", "o", "o", "o", "*"]

        plt.figure(figsize=(8, 8))

        for cls, color, mark, label in zip(
            range(len(colors)), colors, shapes, class_labels
        ):
            mask = self.__features_stress["class"] == cls
            plt.scatter(
                x_error_scaled[mask],
                y_uncertainty_scaled[mask],
                c=color,
                label=label,
                marker=mark,
                alpha=0.7,
            )

        plt.xlabel("Scaled Error", fontsize=14)
        plt.ylabel("Scaled Uncertainty", fontsize=14)
        plt.legend(title="Classes", loc="upper right", fontsize=12, title_fontsize=14)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.axis("square")
        plt.grid(alpha=0.5)
        plt.tight_layout()
        if save:
            if not os.path.exists("figures"):
                os.makedirs("figures")
            plt.savefig("figures/stress_plot.pdf", dpi=500)
        if show:
            plt.show()
        plt.close()

    def __scale_to_unit_range(self, y):
        """
        Scales the given values to a unit range.
        """
        y_min, y_max = np.min(y), np.max(y)
        return (y - y_min) / (y_max - y_min)

    def plot_stress_series(self, stress_type=None, max_series=100):
        """
        Plots the stress-inducing time series
        """
        if stress_type is None:
            stress_type = self.__target_name

        stress_ids_map = {
            "errors": self.large_errors_ids,
            "uncertainty": self.large_uncertainty_ids,
            "hubris": self.hubris_ids,
        }
        stress_ids = stress_ids_map.get(stress_type)
        if stress_ids is None:
            raise ValueError(f"Invalid stress type: {stress_type}")

        fig = plot_series(df=self.train_set, ids=stress_ids, max_ids=max_series)
        return fig
