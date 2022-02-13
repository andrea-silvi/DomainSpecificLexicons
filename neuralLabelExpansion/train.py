import torch
import torch.optim as optim
from torch.nn import MSELoss
import numpy as np
from utils_.early_stopping import EarlyStopping
from dataset.SeedDataset import SeedDataset
from neuralLabelExpansion.RegressionNetwork import RegressionModel
import gc
from utils_.utils import timing_wrapper

CHECKPOINT_PATH = "checkpoint.pt"


@timing_wrapper("Regression network training")
def train(dataset: SeedDataset, batch_size=32, n_workers=2, lr=1e-3, n_epochs=100):
    """
    Trains the regression network given the seed dataset as input.
    """
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
            optimizer.zero_grad()
            wvs = wvs.cuda()
            scores = scores.reshape(-1, 1)
            scores = scores.cuda()
            prediction = model(wvs)
            prediction = prediction.cuda()
            batch_loss = loss(prediction, scores)
            losses.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()
        epoch_loss = np.mean(np.array(losses))
        early_stopping(epoch_loss, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        gc.collect()
        torch.cuda.empty_cache()
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    return model


@timing_wrapper("Regression network prediction")
def predict(model, test_dataset):
    """
    Expands label over to the test dataset of non-seed words.
    """
    model.cuda()
    model.eval()
    results = {}
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2)
    with torch.no_grad:
        for wv, w in test_dataloader:
            wv = wv.cuda()
            pred = model(wv)
            for word, score in zip(w, pred.cpu().squeeze().tolist()):
                results[word] = score
    return results
