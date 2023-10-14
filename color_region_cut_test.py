import cv2
import os
import numpy as np

cascade_path = 'cascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
# 이미지 파일이 있는 디렉토리 경로 설정
image_dir = 'results/'

# 디렉토리 내의 모든 파일 목록 가져오기
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# 이미지를 저장할 리스트 초기화
images = []

# 이미지 파일들을 읽어서 리스트에 저장
for image_file in image_files:
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(image_dir, image_file)
        OriginalImage = cv2.imread(image_path)
        if OriginalImage is not None:
            images.append(OriginalImage)

# images 리스트에는 "img" 폴더 내의 모든 이미지가 저장됩니다.
i = 0
# loading the test image
lower = np.array([0,133,77], dtype = np.uint8)
upper = np.array([255,173,127], dtype = np.uint8)
for image in images:
  i = i + 1
  face_img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
  skin_msk = cv2.inRange(face_img_ycrcb, lower, upper)
  skin = cv2.bitwise_and(image, image, mask = skin_msk)
  save_path = f"colorRegionResults/test{i}.jpg"
  cv2.imwrite(save_path, skin)