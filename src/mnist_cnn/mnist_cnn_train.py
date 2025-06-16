import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


EPOCHS = 5


def train(hyper_cfg, X_train, y_train, X_val, y_val, *, device="cpu"):
    print("Training CNN with hyperparameters:", hyper_cfg)

    class CNN(nn.Module):
        def __init__(self, num_classes=10, conv_channels=[32, 64], kernel_size=3, dropout_rate=0.25):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, conv_channels[0], kernel_size=kernel_size, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=kernel_size, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

            # 👇 Cálculo automático del output flatten size
            with torch.no_grad():
                dummy = torch.zeros(1, 1, 28, 28)
                dummy_out = self.conv_layers(dummy)
                conv_out_size = dummy_out.numel()

            self.flatten = nn.Flatten()
            self.fc_layers = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(conv_out_size, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = x.view(-1, 1, 28, 28)
            x = self.conv_layers(x)
            x = self.flatten(x)
            return self.fc_layers(x)

    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=hyper_cfg.get("batch_size", 64), shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=hyper_cfg.get("batch_size", 64))

    model = CNN(
        conv_channels=[(hyper_cfg.get("conv_channels_1",32)),(hyper_cfg.get("conv_channels_2",64))],
        kernel_size=hyper_cfg.get("kernel_size", 3),
        dropout_rate=hyper_cfg.get("dropout_rate", 0.25),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_cfg.get("learning_rate", 1e-3), weight_decay=hyper_cfg.get("weight_decay", 0.0))
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    for epoch in tqdm(range(EPOCHS), desc="Training Epochs"):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # print(f"Epoch {epoch+1}, Train loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for val_x, val_y in valid_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            outputs = model(val_x)
            preds = outputs.argmax(dim=1)
            correct += (preds == val_y).sum().item()
            total += val_y.size(0)
    acc = correct / total
    print(f"Validation accuracy: {acc:.4f}")
    return acc