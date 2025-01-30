import numpy as np


class PrepareDataset:
    """
    Class to load and prepare datasets for forecasting.
    """

    def __init__(self, data=None):
        self.train = data
        self.valid = None
        self.dev_set = None
        # convert "ds" column to int if not a datetime
        if self.train is not None and isinstance(
            self.train["ds"].iloc[0], (int, np.int32, np.int64)
        ):
            self.train["ds"] = self.train["ds"].astype(int)

    def split_train_set(self, val_size):
        """
        Split the provided train set into development and validation sets based on the split criteria.
        """
        self.valid = self.train.groupby("unique_id").tail(val_size)
        self.dev_set = self.train.drop(self.valid.index).reset_index(drop=True)
        self.train["unique_id"] = self.train["unique_id"].astype("category")
        self.dev_set["unique_id"] = self.dev_set["unique_id"].astype("category")
        self.valid["unique_id"] = self.valid["unique_id"].astype(
            self.dev_set["unique_id"].dtype
        )
