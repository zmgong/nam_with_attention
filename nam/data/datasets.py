from typing import Dict

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder

from nam.config import defaults
from nam.data.base import NAMDatasetOld
from nam.data.folded import FoldedDataset

cfg = defaults()


def load_breast_data(config=cfg) -> Dict:
    breast_cancer = load_breast_cancer()
    dataset = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    dataset['target'] = breast_cancer.target

    config.regression = False

    if config.cross_val:
        return FoldedDataset(config,
                             data_path=dataset,
                             features_columns=dataset.columns[:-1],
                             targets_column=dataset.columns[-1])
    else:
        return NAMDataset(config,
                          data_path=dataset,
                          features_columns=dataset.columns[:-1],
                          targets_column=dataset.columns[-1])


def load_sklearn_housing_data(config=cfg) -> Dict:
    housing = sklearn.datasets.fetch_california_housing()

    dataset = pd.DataFrame(data=housing.data, columns=housing.feature_names)
    dataset['target'] = housing.target

    config.regression = True

    if config.cross_val:
        return FoldedDataset(config,
                             data_path=dataset,
                             features_columns=dataset.columns[:-1],
                             targets_column=dataset.columns[-1])
    else:
        return NAMDataset(config,
                          data_path=dataset,
                          features_columns=dataset.columns[:-1],
                          targets_column=dataset.columns[-1])


def load_housing_data(config=cfg,
                      data_path: str = 'data/housing.csv',
                      features_columns: list = [
                          'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                          'households', 'median_income'
                      ],
                      targets_column: str = 'median_house_value') -> Dict:

    config.regression = True
    feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'population', 'AveOccup', 'Latitude', 'Longitude']

    data = pd.read_csv(data_path)

    data['MedInc'] = data['median_income']
    data['HouseAge'] = data['housing_median_age']
    data['Latitude'] = data['latitude']
    data['Longitude'] = data['longitude']

    # avg rooms = total rooms / households
    data['AveRooms'] = data['total_rooms'] / data["households"]

    # avg bed rooms = total bed rooms / households
    data['AveBedrms'] = data['total_bedrooms'] / data["households"]

    # avg occupancy = population / households
    data['AveOccup'] = data['population'] / data['households']

    data[targets_column] = data[targets_column] / 100000.0

    if config.cross_val:
        return FoldedDataset(config, data_path=data, features_columns=feature_names, targets_column=targets_column)
    else:
        return NAMDataset(config, data_path=data, features_columns=feature_names, targets_column=targets_column)


def load_gallup_data(config=cfg,
                     data_path: str = 'data/GALLUP.csv',
                     features_columns: list = ["country", "income_2", "WP1219", "WP1220", "year", "weo_gdpc_con_ppp"],
                     targets_column: str = "WP16",
                     weights_column: str = "wgt") -> Dict:

    config.regression = False
    data = pd.read_csv(data_path)
    data["WP16"] = np.where(data["WP16"] < 7, 0, 1)
    # data = data.sample(frac=0.1)

    if config.cross_val:
        return FoldedDataset(config,
                             data_path=data,
                             features_columns=features_columns,
                             targets_column=targets_column,
                             weights_column=weights_column)
    else:
        return NAMDataset(config,
                          data_path=data,
                          features_columns=features_columns,
                          targets_column=targets_column,
                          weights_column=weights_column)


def load_compas_data(
    config,
    path: str = '~/nam/data/compas/recid.data',
    features_columns: list = [
        "age", "race", "sex", "priors_count", "length_of_stay", "c_charge_degree"
    ],
    targets_columns: str = ["two_year_recid"],
) -> Dict:
  dataset = pd.read_csv(path, delimiter=' ', header=None)
  dataset.columns = features_columns + targets_columns

  config.regression = False

  if config.cross_val:
      return FoldedDataset(config,
                              data_path=dataset,
                              features_columns=dataset.columns[:-1],
                              targets_column=dataset.columns[-1])
  else:
      return NAMDatasetOld(config,
                              data_path=dataset,
                          features_columns=dataset.columns[:-1],
                          targets_column=dataset.columns[-1])


def load_mtl_compas_data(
    config,
    path: str = '~/nam/data/compas/recid.data',
    features_columns: list = [
        "age", "race", "priors_count", "length_of_stay", "c_charge_degree"
    ],
    targets_columns: str = ["two_year_recid"],
    weights_column: str = ['sex']
) -> Dict:
  dataset = pd.read_csv(path, delimiter=' ', header=None)
  dataset.columns = ["age", "race", "sex", "priors_count", "length_of_stay", "c_charge_degree", "two_year_recid"]
  ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
  new_vals = ohe.fit_transform(dataset[weights_column].to_numpy().reshape(-1, 1))
  new_columns = ohe.categories_[0]
  
  weights_df = pd.DataFrame(new_vals, columns=new_columns)

  dataset = pd.concat([dataset, weights_df], axis=1)

  config.regression = False

  if config.cross_val:
      return FoldedDataset(config,
                              data_path=dataset,
                              features_columns=features_columns,
                              targets_column=targets_columns,
                              weights_columns=new_columns)
  else:
      return NAMDataset(config,
                              data_path=dataset,
                              features_columns=features_columns,
                              targets_column=targets_columns,
                              weights_columns=new_columns)