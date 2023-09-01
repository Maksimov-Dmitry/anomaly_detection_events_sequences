from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig
from typing import List


@dataclass()
class MainParams:
    n_persons: int
    n_train: int
    n_test: int
    nhid: int
    context_intensity_pois_min: List[float]
    context_intensity_pois_factors: List[int]
    non_regulator_outliers_prob: float


MainParamsSchema = class_schema(MainParams)


def read_main_params(cfg: DictConfig) -> MainParams:
    schema = MainParamsSchema()
    return schema.load(cfg)
