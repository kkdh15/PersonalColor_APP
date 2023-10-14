from rembg import remove
from PIL import Image

input = Image.open('./result/test2Cut.jpg') # load image
output = remove(input) # remove background
output.save('rembg.png') # save image