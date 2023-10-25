from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig
from typing import List


@dataclass()
class DatasetParams:
    """
    Parameters for dataset creation.

    Attributes:
        seed (int): Seed to control randomness.

        process (str): Type of process for dataset generation.
                       Currently, only the Poisson process is implemented.

        t_max (int): Maximum time for creating data points.
                     A greater value will result in more data points.

        dt (float): Time interval for generating points.
                    A smaller value will produce denser points.

        n_persons (int): Number of individuals for whom personalized sequences will be created.

        n_train (int): Number of sequences per person to be used for training.

        n_test (int): Number of sequences per person to be used for testing.

        context_intensity_pois_min (List[float]): List of minimum values to create the cumulative
                                                 intensity function for each context. The cumulative intensity
                                                 function is generated from a uniform distribution, ranging
                                                 from `context_intensity_pois_min` to
                                                 `context_intensity_pois_min * context_intensity_pois_factors`.
             
        context_intensity_pois_factors (List[int]): List of factors to determine the upper bound for
                                                    `context_intensity_pois_min` when creating the cumulative
                                                    intensity function.

        non_regulator_outliers_prob (float): Probability to create Comissions or Omissions outliers.
    """
    seed: int
    process: str
    t_max: int
    dt: float
    n_persons: int
    n_train: int
    n_test: int
    context_intensity_pois_min: List[float]
    context_intensity_pois_factors: List[int]
    non_regulator_outliers_prob: float


DatasetParamsSchema = class_schema(DatasetParams)


def read_dataset_params(cfg: DictConfig) -> DatasetParams:
    schema = DatasetParamsSchema()
    return schema.load(cfg)
