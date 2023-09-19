import hydra
from omegaconf import DictConfig
from dataclasses import asdict
from src.entities.main_params import read_main_params, MainParams
import subprocess
import os
from aim import Run, Image
from src.metrics.calculate_metrics import get_metrics
import pickle
from src.visualization.visualize import plot_data


def get_data_plot():
    with open('data/raw/pois/train.pkl', 'rb') as f:
        data_train = pickle.load(f)
        fig = plot_data(data_train[0])
        return fig


def main(params: MainParams):
    run = Run(experiment='main')
    run.add_tag('num-of-sequences-and-seed')
    run['hparams'] = asdict(params)
    create_dataset_command = "python -m src.data.make_dataset "
    for key, value in asdict(params).items():
        if key in ('nhid', 'use_personalisation', 'use_context'):
            continue
        if isinstance(value, list):
            value = f"[{','.join(map(str, value))}]"
        create_dataset_command += f"{key}={value} "
    subprocess.run(create_dataset_command, shell=True, check=True, executable="/bin/bash")
    subprocess.run(f"python src/models/baselines.py --p {params.non_regulator_outliers_prob}", shell=True, check=True, executable="/bin/bash")
    filelist = [f for f in os.listdir("models") if f.endswith(".ckpt")]
    for f in filelist:
        os.remove(os.path.join("models", f))
    fig = get_data_plot()
    run.track(Image(fig), name='initial_data')
    subprocess.run(f"python -m src.models.train_model nhid={params.nhid} use_personalisation={params.use_personalisation} n_persons={params.n_persons} use_context={params.use_context}",
                    shell=True, check=True, executable="/bin/bash")
    model = [f for f in os.listdir("models") if f.endswith(".ckpt")][0]
    os.rename(f'models/{model}', 'models/model.ckpt')
    subprocess.run(f"python -m src.models.predict_model model_path=models/model.ckpt nhid={params.nhid} use_personalisation={params.use_personalisation} n_persons={params.n_persons} use_context={params.use_context} p={params.non_regulator_outliers_prob}",
                    shell=True, check=True, executable="/bin/bash")
    figures, results = get_metrics(params.non_regulator_outliers_prob)
    for fig, _, outlier in figures:
        run.track(Image(fig), name='roc_curve', context={'outliers': outlier})
    for result in results:
        run.track(result['value'], name='roc_auc_score', context={'model': result['method'], 'outliers': result['outlier']})


@hydra.main(version_base=None, config_path="configs", config_name="main_config")
def create_main_command(cfg: DictConfig):
    params = read_main_params(cfg)
    main(params)


if __name__ == '__main__':
    create_main_command()
