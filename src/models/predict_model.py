import pickle
from src.data.dataloader import get_cppod_dataloader
import hydra
from omegaconf import DictConfig
from src.entities.test_params import read_test_params, TestParams
from src.models.model_manager import ModelManager
import torch
import os


def test(params: TestParams):
    number_of_persons = params.n_persons if params.use_personalisation else None
    model = ModelManager(params.lr, params.label_size, params.nhid, params.target, number_of_persons)  # Initialize model with the same parameters as during training
    checkpoint = torch.load(params.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    outlier_types = ['commiss', 'omiss']
    for outlier in outlier_types:
        with open(f'data/raw/{params.process}/test_{outlier}_{params.p}.pkl', 'rb') as f:
            test_data = pickle.load(f)

        test_gen = get_cppod_dataloader(test_data, params.target, 1, params.use_context)
        result = model.predict(test_gen, outlier)
        path = f'results/{params.process}/{outlier}'
        os.makedirs(path, exist_ok=True)
        result.to_csv(f'{path}/CPPOD_{params.p}.csv')


@hydra.main(version_base=None, config_path="../../configs", config_name="predict_config")
def create_test_command(cfg: DictConfig):
    params = read_test_params(cfg)
    test(params)


if __name__ == '__main__':
    create_test_command()
