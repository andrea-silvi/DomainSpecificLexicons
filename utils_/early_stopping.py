import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if training loss doesn't improve after a given patience."""

    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.loss_min = np.Inf

    def __call__(self, loss, model, epoch):

        score = -loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'epoch {epoch}: \n\tearly stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'epoch {epoch}:\n\tLoss decreased ({self.loss_min:.6f} --> {loss:.6f}).  '
                            f'Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.loss_min = loss
