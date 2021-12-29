import torch
from utils.early_stopping import EarlyStopping
from torch.optim import optim
from torch.nn import MSELoss
from SeedDataset import SeedDataset
from RegressionNetwork import RegressionModel

CHECKPOINT_PATH = "checkpoint.pt"

def train(dataset : SeedDataset, *args):
    torch.manual_seed(11)
    loss = MSELoss()
    estop = EarlyStopping(verbose=True, path=CHECKPOINT_PATH)
    estop()


