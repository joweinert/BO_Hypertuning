import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_mnist(filepath: str = '../data/mnist.pkl') -> tuple:

    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')


    (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = data

    # 0 = black , 255 is white
    # -> normal,izing to [0, 1]

    X_train = torch.tensor(train_X / 255.0, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.long)

    X_valid = torch.tensor(valid_X / 255.0, dtype=torch.float32)
    y_valid = torch.tensor(valid_y, dtype=torch.long)

    X_test = torch.tensor(test_X / 255.0, dtype=torch.float32)
    y_test = torch.tensor(test_y, dtype=torch.long)

    return (X_train, y_train, X_valid, y_valid, X_test, y_test)


def load_mnist_for_pytorch_datasets(filepath: str = '../data/mnist.pkl') -> tuple:
    (X_train, y_train, X_valid, y_valid, X_test, y_test) = load_mnist(filepath)

    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset  = TensorDataset(X_test, y_test)

    return (train_dataset, valid_dataset, test_dataset)


def load_mnist_for_pytorch_dataloader(filepath: str = '../data/mnist.pkl', batch_size: int = 64) -> tuple:
    (train_dataset, valid_dataset, test_dataset) = load_mnist_for_pytorch_datasets(filepath)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    return (train_loader, valid_loader, test_loader)

