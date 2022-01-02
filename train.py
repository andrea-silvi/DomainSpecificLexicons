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


def train(dataset: SeedDataset, batch_size=32, n_workers=2, lr=1e-3, n_epochs=100, *args):
    torch.manual_seed(11)
    loss = MSELoss()
    early_stopping = EarlyStopping(verbose=True, path=CHECKPOINT_PATH)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(n_workers))
    model = RegressionModel(low=dataset.get_min_score(), high=dataset.get_max_score())
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.cuda()
    for epoch in range(n_epochs):
        losses = list()
        for wvs, scores in train_dataloader:
            wvs.cuda()
            scores.cuda()
            prediction = model(wvs)
            prediction.cuda()
            batch_loss = loss(prediction, scores)
            losses.append(batch_loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss = np.mean(np.array(losses))
        early_stopping(epoch_loss, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        gc.collect()
        torch.cuda.empty_cache()
        return model.load_state_dict(torch.load(CHECKPOINT_PATH))
        # print(f'\tepoch: {epoch}, training loss: {epoch_loss}')


def predict(model, test_dataset):
    model.cuda()
    model.eval()
    results = {}
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2)
    for wv, w in test_dataloader:
        wv.cuda()
        pred = model(wv)
        results[w] = pred
    return results
