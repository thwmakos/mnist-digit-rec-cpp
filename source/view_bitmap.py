# quick script to read the training image files
# and display a bitmap for testing

from PIL import Image
import numpy as np
import struct
from pathlib import Path

file = "./data/t10k-images-idx3-ubyte"

with open(file, 'rb') as f:
    tdata = np.fromfile(f, dtype=np.uint8)

offset = 16 # image data begin at this offset
image_index = 333 # change this to display the training image at the given index
image_size = 28 * 28 # how many bytes for each image

data_begin = offset + image_index * image_size
data_end = data_begin + image_size

image_array = tdata[data_begin:data_end] # grab the image bitmap

# create PIL image interpreting the data at 28x28
image = Image.frombytes(mode = "L", size = (28, 28), data = image_array)

# display the image
image.show()
