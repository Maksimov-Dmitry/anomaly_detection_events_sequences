from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig
from typing import List


@dataclass()
class MainParams:
    seed: int
    n_persons: int
    n_train: int
    n_test: int
    t_max: int
    dt: float
    nhid: int
    context_intensity_pois_min: List[float]
    context_intensity_pois_factors: List[int]
    non_regulator_outliers_prob: float
    use_personalisation: bool
    use_context: bool


MainParamsSchema = class_schema(MainParams)


def read_main_params(cfg: DictConfig) -> MainParams:
    schema = MainParamsSchema()
    return schema.load(cfg)
