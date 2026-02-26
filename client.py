import argparse
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -------------------------
# Simple CNN for CIFAR-10
# -------------------------
class CIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Load CIFAR-10
# -------------------------
def load_cifar():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    return train, test


# -------------------------
# IID Partition
# -------------------------
def iid_partition(n, num_clients, cid):
    rng = np.random.default_rng(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return splits[cid]


# -------------------------
# Non-IID Partition (Label Based)
# -------------------------
def noniid_partition(dataset, num_clients, cid, labels_per_client=3):
    targets = np.array(dataset.targets)
    all_labels = np.arange(10)

    rng = np.random.default_rng(123)
    rng.shuffle(all_labels)

    start = (cid * labels_per_client) % len(all_labels)
    chosen = all_labels[start:start + labels_per_client]

    if len(chosen) < labels_per_client:
        chosen = np.concatenate(
            [chosen, all_labels[:labels_per_client - len(chosen)]]
        )

    idx = np.where(np.isin(targets, chosen))[0]
    rng2 = np.random.default_rng(999 + cid)
    rng2.shuffle(idx)

    return idx[:15000]


# -------------------------
# DataLoaders
# -------------------------
def get_loaders(cid, num_clients, batch_size, noniid):
    trainset, testset = load_cifar()

    if noniid:
        train_idx = noniid_partition(trainset, num_clients, cid)
    else:
        train_idx = iid_partition(len(trainset), num_clients, cid)

    test_idx = iid_partition(len(testset), num_clients, cid)

    train_loader = DataLoader(
        Subset(trainset, train_idx),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        Subset(testset, test_idx),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


# -------------------------
# Train
# -------------------------
def train(model, loader, device, epochs=1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()


# -------------------------
# Evaluate
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        total_loss += loss.item() * x.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


# -------------------------
# Flower Client
# -------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients, batch_size, noniid):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CIFARNet().to(self.device)

        self.train_loader, self.test_loader = get_loaders(
            cid, num_clients, batch_size, noniid
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        keys = list(state_dict.keys())
        for k, v in zip(keys, parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, self.device, epochs=1)
        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = evaluate(self.model, self.test_loader, self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--noniid", action="store_true")

    args = parser.parse_args()

    client = FlowerClient(
        args.cid,
        args.num_clients,
        args.batch_size,
        args.noniid,
    )

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )


if __name__ == "__main__":
    main()