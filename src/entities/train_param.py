from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig


@dataclass()
class TrainParams:
    dataset: str
    nhid: int
    lr: float
    epochs: int
    sample_multiplier: int
    target: int
    label_size: int
    train_size: float


DatasetParamsSchema = class_schema(TrainParams)


def read_train_params(cfg: DictConfig) -> TrainParams:
    schema = DatasetParamsSchema()
    return schema.load(cfg)
