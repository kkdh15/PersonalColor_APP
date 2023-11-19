import os
import numpy as np
from torchvision import datasets, transforms

# 경로 설정
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

    return (meanR, meanG, meanB), (stdR, stdG, stdB)


# 데이터셋에 대한 평균 및 표준편차 계산
mean, std = get_mean_std(data_dir)
print("평균:", mean)
print("표준편차:", std)
