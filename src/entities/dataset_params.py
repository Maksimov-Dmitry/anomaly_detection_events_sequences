from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig
from typing import List


@dataclass()
class DatasetParams:
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
