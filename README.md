# mastfm

![fig](mastfm-header.png)

`mastfm` is a package for Meta-learning and data Augmentation for Stress Testing. It provides tools to interpret forecasting models by leveraging meta-learning techniques and various data augmentation methods.

## Installation

You can install the `mastfm` package using pip:

```sh
pip install mastfm
```

## Requirements

- numpy
- pandas
- scikit-learn
- shap
- matplotlib
- imbalanced-learn
- xgboost
- lightgbm
- tqdm
- mlforecast

## Usage

Here is a simple example of how to use the `mastfm` package:

```python
import pandas as pd
from mastfm import MAST
from xgboost import XGBRegressor as xgb

# Load your data into a pandas DataFrame
df = pd.read_csv('your_dataset.csv')

# Initialize the MAST class
mast = MASTFM(
    forecasting_model=xgb(),
    seasonality=12,
    frequency="M",
    horizon=12,
    level=80,
    quantile=80,
    augmentation_method="ADASYN",
)
mast.fit(df=df, target_differences=1)
mast.plot_stress(save=True)
mast.explanations(save=True)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Ricardo Inácio