from rembg import remove
from PIL import Image

input = Image.open('./results/cut1.jpg') # load image
output = remove(input) # remove background
output.save('rembg.png') # save image