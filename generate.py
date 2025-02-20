import numpy as np
from PIL import Image

size = (1081, 1081)
height_data = np.linspace(0, 65535, num=size[0] * size[1], dtype=np.uint16).reshape(size)
img = Image.fromarray(height_data, mode='I;16')
img.save("heightmap.png")

