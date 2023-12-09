import torch
import timm
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import matplotlib.pyplot as plt


def main():
    # 데이터 변환
    data_dir = "C:/Users/admin/Downloads/data"

    # 평균 및 표준편차를 계산하는 함수
    def get_mean_std(data_dir):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = datasets.ImageFolder(os.path.join(f'{data_dir}'), transform)
        print("데이터 정보", dataset)

        meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset]
        stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset]

        meanR = np.mean([m[0] for m in meanRGB])
        meanG = np.mean([m[1] for m in meanRGB])
        meanB = np.mean([m[2] for m in meanRGB])

        stdR = np.mean([s[0] for s in stdRGB])
        stdG = np.mean([s[1] for s in stdRGB])
        stdB = np.mean([s[2] for s in stdRGB])

        return [meanR, meanG, meanB], [stdR, stdG, stdB]

    # 데이터셋에 대한 평균 및 표준편차 계산
    mean, std = get_mean_std(data_dir)
    print("평균:", mean)
    print("표준편차:", std)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    models = {
        'mobilenetv3_large_100': timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=9).to(device),
        'efficientnet_b0': timm.create_model('efficientnet_b0', pretrained=False, num_classes=9).to(device),
        'densenet': timm.create_model('densenet121', pretrained=False, num_classes=9).to(device),
    }

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir), train_transform)

    # 데이터셋 분할
    valid_size = 0.3
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=valid_sampler, num_workers=4)

    def train_model(model, train_loader, valid_loader, epochs=50, early_stopping_rounds=15):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        train_losses, valid_losses = [], []
        train_accuracies, valid_accuracies = [], []

        best_val_loss = np.inf
        loss_increase_count = 0

        for epoch in range(epochs):
            model.train()
            running_loss, running_corrects, num_samples = 0.0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                num_samples += labels.size(0)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / num_samples
            epoch_acc = running_corrects.double() / num_samples

            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            # 검증 단계
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

            epoch_val_loss = valid_loss / num_samples
            epoch_val_acc = valid_corrects.double() / num_samples

            valid_losses.append(epoch_val_loss)
            valid_accuracies.append(epoch_val_acc)

            print(
                f'Epoch {epoch + 1}/{epochs} - Train loss: {epoch_loss:.4f}, Valid loss: {epoch_val_loss:.4f}, Train acc: {epoch_acc:.4f}, Valid acc: {epoch_val_acc:.4f}')


            # Early stopping logic
            # if epoch_val_loss < best_val_loss:
            #   best_val_loss = epoch_val_loss
            #    loss_increase_count = 0
            #else:
            #    loss_increase_count += 1
            #    if loss_increase_count == early_stopping_rounds:
            #        print(f'Early stopping triggered at epoch {epoch + 1}')
            #        break


        return train_losses, valid_losses, train_accuracies, valid_accuracies


    results = {}
    for name, model in models.items():
        print(f'Training {name}...')
        train_losses, valid_losses, train_accuracies, valid_accuracies = train_model(model, train_loader, valid_loader)
        results[name] = {
            'train_loss': train_losses,
            'valid_loss': valid_losses,
            'train_acc': train_accuracies,
            'valid_acc': valid_accuracies
        }

    for name, result in results.items():
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(result['train_loss'], label='Train Loss')
        plt.plot(result['valid_loss'], label='Valid Loss')
        plt.title(f'{name} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(result['train_acc'], label='Train Accuracy')
        plt.plot(result['valid_acc'], label='Valid Accuracy')
        plt.title(f'{name} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
