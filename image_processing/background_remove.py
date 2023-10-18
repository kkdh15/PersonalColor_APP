from rembg import remove
from PIL import Image
# i = 0
# for j in range(12):
# 	i += 1
# 	input = Image.open(f"./results/cut{i}.jpg") # load image
# 	output = remove(input) # remove background
# 	output.save(f'./no_background/{i}.png') # save image

input = Image.open(f"./results/cut12.jpg") # load image
output = remove(input) # remove background
output.save(f'./no_background/13.png') # save image