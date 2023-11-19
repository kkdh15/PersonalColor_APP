import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

def main():
    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 경로 설정
    data_dir = "C:/Users/admin/Downloads/data"

    # 데이터 변환 설정
    train_transform = transforms.Compose([

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5439718, 0.49177063, 0.48634678], [0.2782378, 0.26515168, 0.26143256])
])
    #         transforms.RandomHorizontalFlip(),  # 좌우 반전
    #         transforms.RandomVerticalFlip(),  # 상하반전

    # 데이터셋 로드
    train_dataset = datasets.ImageFolder(os.path.join(data_dir), train_transform)

    # trainning set 중 test 데이터로 사용할 비율
    valid_size = 0.3
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # trainning, test batch를 얻기 위한 sample
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=32,
                                               sampler=train_sampler,
                                               num_workers=4)

    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=32,
                                               sampler=valid_sampler,
                                               num_workers=4)

    # timm을 통해 EfficientNet B0 모델 로드 (pretrained=False)
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=9)  # 클래스 레이블이 9개라고 가정
    model.to(device)

    # 손실 함수 및 최적화함수 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 학습 함수 정의
    def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=10):
        history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}

        for epoch in range(num_epochs):
            # 학습 단계
            model.train()
            train_loss = 0.0
            train_corrects = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                train_loss += loss.item() * inputs.size(0)
                train_corrects += torch.sum(preds == labels.data)

            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_corrects.double() / len(train_loader.dataset)

            # 검증 단계
            model.eval()  # 모델을 평가 모드로 설정
            test_loss = 0.0
            test_corrects = 0

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    test_loss += loss.item() * inputs.size(0)
                    test_corrects += torch.sum(preds == labels.data)

            test_loss = test_loss / len(valid_loader.dataset)
            test_acc = test_corrects.double() / len(valid_loader.dataset)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            print(
                f'Epoch {epoch}/{num_epochs - 1} Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

        return history

    def show_images(images, labels, class_names):
        plt.figure(figsize=(15, 7))
        for i in range(len(images)):
            plt.subplot(1, len(images), i + 1)
            img = images[i].numpy().transpose((1, 2, 0))
            #mean = np.array([0.5439718, 0.49177063, 0.48634678])
            #std = np.array([0.2782378, 0.26515168, 0.26143256])
            #img = std * img + mean
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            plt.title(class_names[labels[i]])
            plt.axis('off')
        plt.show()

    # 데이터셋에서 몇 개의 이미지를 불러오기
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # 이미지와 레이블을 함께 시각적으로 출력
    show_images(images[:5], labels[:5], train_dataset.classes)

    print(train_dataset.classes)

    # 학습 시작
    model_trained = train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=100)

    # 모델 저장
    torch.save(model_trained[0].state_dict(), 'efficientnet_b0_model.pth')


# 메인 함수 실행
if __name__ == '__main__':
    history = main()  # main 함수 수정
    # 학습과 검증의 손실 그래프
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['valid_loss'], label='Valid Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 학습과 검증의 정확도 그래프
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['valid_acc'], label='Valid Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
