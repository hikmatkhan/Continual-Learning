# -*- coding: utf-8 -*-
"""
Copyright 2021 Christopher Francis Angelini

"""

# Base Python
import os
import time
# PyTorch
import torch
# PyTorch Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import kde_plot_layers

# Tensorboard
# Remote: ssh -NfL localhost:8898:localhost:6000 user@remote.
# $tensorboard --logdir=./logs --port=6000


def run(args, l_model, train_loader, test_loader):
    the_time = time.time()
    start_date_str = time.strftime("%m_%d_%y", time.localtime(the_time))
    start_time_str = time.strftime("%H_%M", time.localtime(the_time))
    if args.wandb:
        wandb_logger = WandbLogger(name='Run_' + start_date_str + '_' + start_time_str, project=args.project, entity=args.account)
        wandb_logger.log_hyperparams(args)
        wandb_logger.watch(l_model.model, log='gradients', log_freq=100)
    else:
        wandb_logger = None

    # Define model checkpoint paths
    dir_val_path = args.base_save + 'val_checkpoints/'
    dir_val_name = 'model-{epoch:02d}-{valid_acc:.2f}'

    if args.kdes:
        kde_path = args.base_save + 'KDEs/'
        if not os.path.exists(kde_path):
            os.makedirs(kde_path)
        kde_plot_layers(l_model.model, 0, kde_path)


    # Validation checkpoint
    val_checkpoint_callback = ModelCheckpoint(
        monitor='train_acc',
        dirpath=dir_val_path,
        filename=dir_val_name,
        save_top_k=3,
        mode='max')

    if torch.cuda.device_count() == 0:
        print('You\'re gonna need a gpu...')
        return
    elif torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(str() + ' devices available')
        backend = 'ddp'
        pl_plugins = DDPPlugin(find_unused_parameters=False)
    else:
        num_gpus = torch.cuda.device_count()
        print('Using device ' + str(torch.cuda.get_device_name()))
        backend = None
        pl_plugins = None

    # Train or Continue Train
    if args.load_model != '':

        # automatically restores model, epoch, step, LR schedulers, apex, etc...
        trainer = Trainer(accelerator=backend, resume_from_checkpoint=args.load_model,
                          max_epochs=args.epochs, gpus=num_gpus,
                          callbacks=[val_checkpoint_callback],
                          logger=wandb_logger, plugins=pl_plugins)

        if args.continue_train:
            trainer.fit(l_model, train_dataloader=train_loader, val_dataloaders=test_loader)

        trainer.test(l_model, test_dataloaders=test_loader)
    else:
        trainer = Trainer(distributed_backend=backend,
                          max_epochs=args.epochs, gpus=num_gpus,
                          callbacks=[val_checkpoint_callback],
                          logger=wandb_logger, plugins=pl_plugins, stochastic_weight_avg=True)

        trainer.fit(l_model, train_dataloader=train_loader, val_dataloaders=test_loader)
        trainer.save_checkpoint(args.base_save + '/model-end-training.ckpt')

        trainer.test(l_model, test_dataloaders=test_loader)

    return None