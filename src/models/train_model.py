import pickle
from src.data.dataloader import get_cppod_dataloader
import hydra
from omegaconf import DictConfig
from src.entities.train_param import read_train_params, TrainParams
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from src.models.model_manager import ModelManager


def train(params: TrainParams):
    with open(params.dataset, 'rb') as f:
        data_train = pickle.load(f)

    n_train = int(len(data_train) * params.train_size)
    train = data_train[:n_train]
    val = data_train[n_train:]

    train_gen = get_cppod_dataloader(train, params.target, params.sample_multiplier)
    val_gen = get_cppod_dataloader(val, params.target, params.sample_multiplier)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath='models',
        filename='NSMMPP-{epoch:02d}-{val_loss:.0f}',
        save_top_k=1,
        verbose=True,
        save_last=False,
    )
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    model = ModelManager(params.lr, params.label_size, params.nhid, params.target)
    trainer = pl.Trainer(max_epochs=params.epochs, accelerator="cpu", devices=1, callbacks=[checkpoint_callback, early_stopping],
                         log_every_n_steps=1, logger=False)
    trainer.fit(model, train_dataloaders=train_gen, val_dataloaders=val_gen)


@hydra.main(version_base=None, config_path="../../configs", config_name="train_config")
def create_train_command(cfg: DictConfig):
    params = read_train_params(cfg)
    train(params)


if __name__ == '__main__':
    create_train_command()
