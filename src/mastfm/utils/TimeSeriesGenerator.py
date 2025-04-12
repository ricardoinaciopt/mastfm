from metaforecast.synth.generators.tsmixup import TSMixup
from metaforecast.synth.generators.kernelsynth import KernelSynth
from metaforecast.synth.generators.mbb import SeasonalMBB
from metaforecast.synth.generators.dba import DBA
from metaforecast.synth.generators.jittering import Jittering
from metaforecast.synth.generators.scaling import Scaling
from metaforecast.synth.generators.warping_mag import MagnitudeWarping
from metaforecast.synth.generators.warping_time import TimeWarping
import pandas as pd
import inspect


class TimeSeriesGenerator:
    """
    Unified wrapper for synthetic time series generation methods
    from the MetaForecast package. It enables seamless augmentation of time series
    datasets using various techniques such as mixing, kernel synthesis, magnitude
    warping, jittering, and more.
    """

    def __init__(
        self,
        df,
        seasonality=None,
        frequency=None,
        min_len=None,
        max_len=None,
        target=None,
        quantile=None,
    ):
        """
        Initializes the TimeSeriesGenerator with dataset attributes and pre-configures
        augmentation methods.
        """
        self.df = df
        self.seasonality = seasonality
        self.frequency = frequency
        self.min_len = min_len
        self.max_len = max_len
        self.target = target
        self.quantile = quantile
        self.methods = {
            "TSMixup": [
                TSMixup,
                TSMixup(max_n_uids=3, min_len=self.min_len, max_len=self.min_len),
            ],
            "DBA": [DBA, DBA(max_n_uids=3)],
            "Scaling": [Scaling, Scaling()],
            "MagnitudeWarping": [MagnitudeWarping, MagnitudeWarping()],
            "TimeWarping": [TimeWarping, TimeWarping()],
            "Jittering": [Jittering, Jittering()],
        }

    def get_class_methods(self, cls):
        """
        Returns the names of all methods defined within a given class.
        """
        methods = inspect.getmembers(cls, predicate=inspect.isfunction)
        class_methods = [
            name
            for name, func in methods
            if func.__qualname__.startswith(cls.__name__ + ".")
        ]
        return class_methods

    def generate_synthetic_dataset(self, method_name):
        """
        Generates a synthetic dataset using the specified augmentation method.
        """
        if not method_name or method_name not in self.methods:
            raise ValueError(f"Unknown method_name: {method_name}")

        gen_diff = self.df[self.target].value_counts().get(0, 0) - self.df[
            self.target
        ].value_counts().get(1, 0)
        if gen_diff > 100000:
            gen_diff //= 10
        if gen_diff > 200000:
            gen_diff //= 50

        if method_name in {"DBA", "TSMixup"}:
            self.df["unique_id"] = self.df["unique_id"].astype("str")

        method = self.methods.get(method_name)[1]
        cls = self.methods.get(method_name)[0]

        df = self.df[self.df[self.target] == 1].drop(columns=self.target)
        augmented_dfs = []

        if "transform" in self.get_class_methods(cls):
            gen_diff //= 60
            try:
                augmented_df = method.transform(df, gen_diff)
                augmented_df["unique_id"] = (
                    augmented_df["unique_id"].astype(str) + "_SYN"
                )
            except ValueError as e:
                if "Expected n_neighbors <= n_samples_fit" in str(e):
                    raise ValueError(
                        "This definition of stress is too rigid, and there are not enough examples to augment the dataset. "
                        f"Please relax the stress threshold ('quantile' < {int(self.quantile*100)}), or change the 'augmentation_method'."
                    )
                else:
                    raise e
            augmented_dfs.append(augmented_df.copy())
        else:
            if gen_diff > 100000:
                gen_diff //= 10
            gen_diff //= 100
            if self.quantile <= 0.9:
                gen_diff //= 80
            for i in range(gen_diff):
                augmented_df = method._create_synthetic_ts(df)
                augmented_df["unique_id"] = (
                    augmented_df["unique_id"].astype(str).str.split("_").str[0]
                    + f"_SYN{i+1}"
                )
                augmented_dfs.append(augmented_df.copy())

        return pd.concat(augmented_dfs, axis=0).reset_index(drop=True)
