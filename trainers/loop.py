from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.networks import NeuralNetwork


def loss_function(
        cfg: dict, outputs: torch.Tensor, label: torch.Tensor
    ) -> torch.nn:
    criteria = {
        "mse": nn.MSELoss(),
        "ce": nn.CrossEntropyLoss()
    }
    criterion = criteria[cfg["criterion"]]
    loss = criterion(outputs, label)
    return loss


def train_nn(
        epochs: int, train_dataloader: DataLoader, valid_dataloader: DataLoader,
        model: NeuralNetwork, optimizer, batch_size: int, device: str, cfg: dict
    ) -> Tuple[NeuralNetwork, list]:
    dataloader_dict = {"Train": train_dataloader, "Valid": valid_dataloader}

    train_data = []
    valid_data = []

    with tqdm(range(epochs)) as pbar_epoch:
        for epoch in pbar_epoch:
            pbar_epoch.set_description(f"epoch : {epoch + 1}")

            for phase in ["Train", "Valid"]:
                if phase == "Train":
                    model.train()
                else:
                    model.eval()

                epoch_loss = 0.0

                for inputs, label in dataloader_dict[phase]:
                    inputs = inputs.to(device)
                    label = label.to(device)

                    with torch.set_grad_enabled(phase == "Train"):
                        output = model(inputs)
                        loss = loss_function(cfg, output, label)

                        if phase == "Train":
                            loss.backward()
                            optimizer.step()

                        epoch_loss += loss.item()

                epoch_loss /= len(dataloader_dict[phase].dataset) * batch_size

                if phase == "Train":
                    train_data.append(epoch_loss)
                else:
                    valid_data.append(epoch_loss)

    data = [{cfg["criterion"]: train_data}, {cfg["criterion"]: valid_data}]
    training_data = dict(zip(["Train", "Valid"], data))

    return model, training_data
