import os
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
from rembg import remove
from PIL import Image

data_dir = ['data/1_spr_light', 'data/2_spr_blight', 'data/3_sum_light', 'data/4_sum_mute',
            'data/5_fal_deep', 'data/6_fal_mute', 'data/7_win_blight', 'data/8_win_deep',
            'data/9_win_pale'
			]
type_of_personal_color = ['1_spr_light', '2_spr_blight', '3_sum_light', '4_sum_mute',
            '5_fal_deep', '6_fal_mute', '7_win_blight', '8_win_deep',
            '9_win_pale'
			]

only_image_name = []
for dir in data_dir:
	only_image_name.append([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])

paths_of_image = []
for j in range(9):
	paths_of_image.append([])
i = 0
for image_files in only_image_name:
	for image in image_files:
		if image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
			image_path = os.path.join(data_dir[i], image)
			OriginalImage = cv2.imread(image_path)
		if OriginalImage is not None:
			paths_of_image[i].append(OriginalImage)
	i += 1
print("파일 불러오기 완료")
i = 0
paths_of_aligned_image = []
for j in range(9):
	paths_of_aligned_image.append([])
for dir in paths_of_image:
	j = 1
	print(type_of_personal_color[i])
	print("------------------------------\n\n")
	for image in dir:
		if image is not None:
			aligned_img = DeepFace.detectFace(image, enforce_detection=False)
			aligned_path = f"face_alignment/{type_of_personal_color[i]}/{j}.png"
			paths_of_aligned_image[i].append(aligned_path)
			plt.imsave(aligned_path, aligned_img)
			print(type_of_personal_color[i], j, "번째 완료")
		else:
			print("이미지 처리 에러")
		j += 1
	i += 1