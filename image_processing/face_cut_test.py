import cv2
import os

cascade_path = 'cascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
# 이미지 파일이 있는 디렉토리 경로 설정
image_dir = 'data/Spr_Blight'

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
for image in images:
  i = i + 1
  if image is not None:
    # 이미지  처리 및 얼굴 감지 코드를 이어서 실행
  # converting to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray)
    # for every face, cut the face region and save it as a new file.
    for x, y, width, height in faces:
      roi = image[y:y+height, x:x+width]

    save_path = f"results/cut{i}.jpg"
    cv2.imwrite(save_path, roi)
  else:
      print("이미지를 불러오는 데 문제가 있습니다. 이미지 파일 경로를 확인하세요.")
      #ㅇㅇ