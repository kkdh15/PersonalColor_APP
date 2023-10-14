import cv2

# loading the test image
image = cv2.imread("imgs/Test2.jpg");
cascade_path = 'cascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if image is not None:
    # 이미지 처리 및 얼굴 감지 코드를 이어서 실행
# converting to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray)
    # print the number of faces detected
    print(f"{len(faces)} faces detected in the image.")
    # for every face, draw a blue rectangle
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
        # save the image with rectangles
    cv2.imwrite("test_detected1.jpg", image)
else:
    print("이미지를 불러오는 데 문제가 있습니다. 이미지 파일 경로를 확인하세요.")
    #ㅇㅇ