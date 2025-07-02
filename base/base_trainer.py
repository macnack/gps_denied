import torch
from abc import abstractmethod
from numpy import inf
import os
import gc

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, logger):
        self.config = config
        self.logger = logger

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.best_valid_loss = float("inf")

        self.start_epoch = 1

        self.checkpoint_dir = config.get('model_path')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        print("PID: ", os.getpid())
        print(f"Run ID: {self.logger['sys/id'].fetch()}")

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            self.logger["train/epoch_loss"].log(result['train'])
            self.logger["train/epoch"] = epoch

            if result['val'] != None:
                if result['val'] < self.best_valid_loss:
                    self._save_checkpoint()
            
            gc.collect()  # garbage collection
            

    def _save_checkpoint(self):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        run_name = self.logger["sys/id"].fetch()
        filename = f"dlk_weights_{run_name}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.conv_func, filepath)
        print(f"New best model saved as {filepath}")



    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        raise NotImplementedError