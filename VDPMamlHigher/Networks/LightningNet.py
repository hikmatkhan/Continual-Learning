# -*- coding: utf-8 -*-
"""
Copyright 2021 Christopher Francis Angelini

"""
# PyTorch
import torch
import torch.nn as nn
# PyTorch Lightning
import pytorch_lightning as pl
from utils import kde_plot_layers


# Tensorboard
# Remote: ssh -NfL localhost:8898:localhost:6000 user@remote.
# $tensorboard --logdir=./logs --port=6000


# Lightning Class for model
class Lightning_Net(pl.LightningModule):
    def __init__(self, model, optim, sched, args):
        """ Initialization of the training class for PyTorch Lightining

        :param learning_rate: float
        :param scheduler: list
        :param gamma: float
        :param tau: float
        :param drop_out: float
        """
        super(Lightning_Net, self).__init__()
        # Logging
        self.save_hyperparameters()
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

        self.out_shape = list(model.children())[-2].out_features

        self.kdes = args.kdes
        self.base_save = args.base_save
        self.kde_path = args.base_save + 'KDEs/'

        self.model = model

        self.optim = optim
        self.sched = sched

    def training_step(self, batch, batch_idx):
        """ Training forward Pass

        :param batch: ([batch, channels, height, width],[batch, num_classes])
        :param batch_idx: int
        :return loss: float
        """
        x, y = batch
        mu, sigma = self.model(x)
        labels = nn.functional.one_hot(y, self.out_shape)
        loss = self.model.batch_loss(mu, sigma, labels)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_accuracy(mu, y))
        return loss

    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        """ Validation forward Pass

        :param batch: ([batch, channels, height, width],[batch, num_classes])
        :param batch_idx: int
        """
        x, y = batch
        mu, sigma = self.model(x)
        #self.log('valid_loss', loss)
        self.log('valid_acc', self.val_accuracy(mu, y))
        return True

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        """ Testing forward Pass

        :param batch: ([batch, channels, height, width],[batch, num_classes])
        :param batch_idx: int
        """
        x, y = batch
        mu, sigma = self.model(x)
        self.zero_grad()
        self.log('test_acc', self.test_accuracy(mu, y))

    def on_train_epoch_end(self, outputs):
        """ Put anything to run at the end of the epoch here

        :param outputs:
        :return None:
        """
        if self.kdes:
            if self.current_epoch > 0 and self.current_epoch % 2 == 0:
                kde_plot_layers(self.model, self.current_epoch, self.kde_path)

        self.log('train_acc_epoch', self.train_accuracy.compute())
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        """ Put anything to run at the end of the epoch here

        :param outputs:
        :return None:
        """
        self.log('val_acc_epoch', self.val_accuracy.compute())
        self.val_accuracy.reset()

    def configure_optimizers(self):
        """ Configure optimizers and schedulers

        :return optimizers: []
        :return schedulers: []
        """
        optimizers = self.optim
        lr_scheduler = self.sched
        return [optimizers], [lr_scheduler]

    def get_progress_bar_dict(self):
        """ Configure progress bar

        :return:
        """
        # don't show the version numbery
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items