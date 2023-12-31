from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig


@dataclass()
class TestParams:
    model_path: str
    process: str
    p: float
    target: int
    label_size: int
    nhid: int
    lr: float
    use_personalisation: bool
    n_persons: int
    use_context: bool


DatasetParamsSchema = class_schema(TestParams)


def read_test_params(cfg: DictConfig) -> TestParams:
    schema = DatasetParamsSchema()
    return schema.load(cfg)
