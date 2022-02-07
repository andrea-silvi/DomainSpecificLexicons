import torch
import torch.optim as optim
from torch.nn import MSELoss
import numpy as np
from utils.early_stopping import EarlyStopping
from SeedDataset import SeedDataset
from RegressionNetwork import RegressionModel
import neptune.new as neptune
import gc
from utils.utils import timing_wrapper

CHECKPOINT_PATH = "checkpoint.pt"

@timing_wrapper("Regression network training")
def train(dataset: SeedDataset, run,  batch_size=32, n_workers=2, lr=1e-3, n_epochs=100, *args):
    torch.manual_seed(11)
    loss = MSELoss()
    early_stopping = EarlyStopping(verbose=True, path=CHECKPOINT_PATH)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(n_workers))
    model = RegressionModel(low=dataset.get_min_score(), high=dataset.get_max_score())
    run["config/model"] = type(model).__name__
    run["config/criterion"] = type(loss).__name__

    optimizer = optim.Adam(model.parameters(), lr=lr)
    run["config/optimizer"] = type(optimizer).__name__
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
        #TODO: neptune epoch loss

        epoch_loss = np.mean(np.array(losses))
        run["training/batch/loss"].log(epoch_loss)
        early_stopping(epoch_loss, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        gc.collect()
        torch.cuda.empty_cache()

        #TODO: save checkpoint su neptune da CHECKPOINT PATH
        run["model_dictionary"].upload(CHECKPOINT_PATH)

    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    return model
        # print(f'\tepoch: {epoch}, training loss: {epoch_loss}')


@timing_wrapper("Regression network prediction")
def predict(model, test_dataset):
    model.cuda()
    model.eval()
    results = {}
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2)
    for wv, w in test_dataloader:
        wv = wv.cuda()
        pred = model(wv)
        for word, score in zip(w, pred.cpu().squeeze().tolist()):
            results[word] = score
    return results
