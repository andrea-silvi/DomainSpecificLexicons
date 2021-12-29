import torch
import torch.optim as optim
from torch.nn import MSELoss
import numpy as np
from utils.early_stopping import EarlyStopping
from SeedDataset import SeedDataset
from RegressionNetwork import RegressionModel
import neptune.new as neptune
import gc

CHECKPOINT_PATH = "checkpoint.pt"


def train(dataset: SeedDataset, batchSize=32, n_workers=2, lr=1e-3, n_epochs=100, *args):
    torch.manual_seed(11)
    loss = MSELoss()
    estop = EarlyStopping(verbose=True, path=CHECKPOINT_PATH)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=int(n_workers))
    network = RegressionModel(low=dataset.get_min_score(), high=dataset.get_max_score())
    optimizer = optim.Adam(network.parameters(), lr=lr)
    network.cuda()
    for epoch in range(n_epochs):
        losses = list()
        for wvs, scores in train_dataloader:
            wvs.cuda()
            scores.cuda()
            prediction = network(wvs)
            prediction.cuda()
            batch_loss = loss(prediction, scores)
            losses.append(batch_loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss = np.mean(np.array(losses))
        gc.collect()
        torch.cuda.empty_cache()
        print(f'\tepoch: {epoch}, training loss: {epoch_loss}')





