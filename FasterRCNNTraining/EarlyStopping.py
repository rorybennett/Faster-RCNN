"""
Patience class to check if training should be cancelled due to no improvement in the validation set.
"""

from os.path import join

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=100, delta=0.05):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_epoch = 0
        # If current score within this range, it is considered the same as previous score.
        self.delta = delta

    def __call__(self, val_loss, model, epoch, optimiser, save_path):
        score = -val_loss

        if self.best_score is None:
            print(f'Saving first model... ', end='')
            self.best_epoch = epoch
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimiser, save_path)
        elif score < self.best_score + self.delta:
            print(f'No improvement (delta: {self.delta})... ', end='')
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print('New best! Saving model... ', end='')
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model, epoch, optimiser, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimiser, save_path):
        """Saves model when validation loss decreases."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
        }, join(save_path, 'model_best.pth'))
        self.val_loss_min = val_loss
