import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


EPOCHS = 5

def train(hyper_cfg, X_train, y_train, X_val, y_val, *, device="cpu"):
    print("Training MLP with hyperparameters:", hyper_cfg)
    class MLP(nn.Module):
        def __init__(self, input_dim=784, num_classes=10, n_layers=2, hidden_dims=None, dropout_rates=None, activations=None, use_layernorm=True):
            super().__init__()
            
            if not isinstance(hidden_dims, list):
                hidden_dims = [hidden_dims] * n_layers
            if not isinstance(dropout_rates, list):
                dropout_rates = [dropout_rates] * n_layers
            if not isinstance(activations, list):
                activations = [activations] * n_layers
            
            layers = []
            current_dim = input_dim

            for i in range(n_layers):
                layers.append(nn.Linear(current_dim, hidden_dims[i]))
                if use_layernorm:
                    layers.append(nn.LayerNorm(hidden_dims[i]))
                if activations[i] is not None:
                    layers.append(activations[i]())
                if dropout_rates[i] is not None and dropout_rates[i] > 0:
                    layers.append(nn.Dropout(dropout_rates[i]))
                current_dim = hidden_dims[i]

            layers.append(nn.Linear(current_dim, num_classes))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    train_dataset, valid_dataset = TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)
    train_loader, valid_loader = DataLoader(train_dataset, batch_size=hyper_cfg.get("batch_size", 64), shuffle=True), DataLoader(valid_dataset, batch_size=hyper_cfg.get("batch_size",64))

    model = MLP(
        hidden_dims=hyper_cfg.get("hidden_dim", 128),
        n_layers=hyper_cfg.get("n_layers", 2),
        dropout_rates= hyper_cfg.get("dropout_rate", 0.0),
        use_layernorm= hyper_cfg.get("use_layernorm", True),
        activations=hyper_cfg.get("activation", nn.ReLU),
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

        #print(f"Epoch {epoch+1}, Train loss: {total_loss / len(train_loader):.4f}") #removed due to SPAAM 

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