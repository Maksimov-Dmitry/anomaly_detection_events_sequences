import torch
import pytorch_lightning as pl
from src.models.cppod import NSMMPP
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm


class ModelManager(pl.LightningModule):
    """
    Model Manager for handling the training and prediction of CPPOD models.

    Attributes:
        model (NSMMPP): Neural Stochastic Modeling of Mixed Point Process.
        lr (float): Learning rate.
        sim_time_diffs (tensor): Differences in simulation times, initially None.
        dt (int): Time difference, default is 1.
    """
    def __init__(self, lr: float, label_size: int, hidden_size: int, target: int, number_of_persons=None):
        super(ModelManager, self).__init__()
        self.model = NSMMPP(label_size, hidden_size, target, number_of_persons)
        self.lr = lr
        self.sim_time_diffs = None
        self.dt = 1

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step(self, batch, batch_idx, metric):
        seq = batch[0]
        loss = self.forward(
            seq['label_seq'],
            seq['time_seq'],
            seq['sim_time_seq'],
            seq['sim_time_idx'],
            seq['id']
        )
        self.log(metric, loss, on_epoch=True, on_step=metric == 'train_loss', prog_bar=True, batch_size=1)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val_loss')

    def predict(self, dataloader, outliers_type):
        """
        Predict the outliers in the given dataloader.

        Args:
            dataloader (DataLoader): DataLoader object.
            outliers_type (str): Type of outliers ('commiss' or 'omissions').

        Returns:
            DataFrame: Results in a DataFrame.
        """
        results = []
        for seq_idx, batch in tqdm(enumerate(dataloader)):
            item = batch[0]
            if outliers_type == 'commiss':
                score_omiss, score_commiss, _, _ = self.model.detect_outlier_instant(
                    item['label_seq'],
                    item['time_seq'],
                    item['sim_time_seq'],
                    item['sim_time_idx'],
                    item['sim_time_diffs'],
                    item['time_test'],
                    item['id'],
                )
            else:
                score_omiss, score_commiss, _, _ = self.model.detect_outlier(
                    item['label_seq'],
                    item['time_seq'],
                    item['sim_time_seq'],
                    item['sim_time_idx'],
                    item['sim_time_diffs'],
                    item['time_test'],
                    item['id'],
                    n_sample=1000,
                )
            df = pd.DataFrame(OrderedDict({
                'seq': seq_idx,
                'time': item['time_test'].numpy()[1:],
                'score_omiss': score_omiss.numpy(),
                'score_commiss': score_commiss.numpy(),
                'label': item['label_test'].numpy(),
            }))
            if item['id'] is None:
                df.insert(0, 'id', seq_idx + 1)
            else:
                df.insert(0, 'id', item['id'])
            results.append(df)
        results = pd.concat(results)
        return results

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return [opt], []
