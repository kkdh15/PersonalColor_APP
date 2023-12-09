import torch
import timm
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, valid_loader, optimizer, criterion, epochs=10, device='cuda'):
    model.to(device)
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss, valid_corrects, num_samples = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                num_samples += labels.size(0)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)
                valid_corrects += torch.sum(preds == labels.data)

        epoch_val_acc = valid_corrects.double() / num_samples
        best_val_acc = max(best_val_acc, epoch_val_acc)
        print(
            f'Epoch {epoch + 1}/{epochs} Valid acc: {epoch_val_acc:.4f}')

    return best_val_acc.item()

# 베이지안 최적화 목적 함수
space = [
    Categorical(['adam', 'sgd'], name='optimizer_name'),
    Categorical([16, 32, 64, 128], name='batch_size'),
]

@use_named_args(space)
def objective(optimizer_name, batch_size, device='cuda'):
    data_dir = "C:/Users/admin/Downloads/data"
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.64334404, 0.52448505, 0.4868242], [0.2552387, 0.23120242, 0.21927562])
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False)

    model = timm.create_model(
        'efficientnet_b0',
        num_classes=len(dataset.classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02) if optimizer_name == 'adam' else optim.SGD(model.parameters(), lr=0.02)

    print(f"{optimizer_name}, batch_size={batch_size} 학습 시작")
    val_accuracy = train_model(model, train_loader, test_loader, optimizer, criterion, epochs=5)
    print(f"Current space: {optimizer_name}, batch_size={batch_size}")
    print(f"Current val_acc: {val_accuracy}")
    return -val_accuracy

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)
    print("Best parameters: {}".format(res_gp.x))


if __name__ == '__main__':
    main()
