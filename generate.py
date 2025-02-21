import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
import noise

size = 1081
mask = np.zeros((size, size), dtype=np.uint8)
np.random.seed(42)
num_circles = 3
for _ in range(num_circles):
    cx = np.random.randint(int(size * 0.3), int(size * 0.7))
    cy = np.random.randint(int(size * 0.3), int(size * 0.7))
    r = np.random.randint(int(size * 0.1), int(size * 0.2))
    y, x = np.ogrid[:size, :size]
    circle = (x - cx)**2 + (y - cy)**2 <= r**2
    mask[circle] = 1

dist = distance_transform_edt(mask)
dist = np.max(dist) - dist
dist = dist * (mask > 0)

noise_scale = 100.0
noise_octaves = 6
noise_persistence = 0.5
noise_lacunarity = 2.0
noise_array = np.zeros((size, size), dtype=np.float32)
for i in range(size):
    for j in range(size):
        noise_array[i][j] = noise.pnoise2(i / noise_scale, j / noise_scale, octaves=noise_octaves, persistence=noise_persistence, lacunarity=noise_lacunarity, repeatx=size, repeaty=size, base=42)
noise_array = (noise_array - noise_array.min()) / (noise_array.max() - noise_array.min())
height = dist + noise_array * np.max(dist) * 0.2
height = gaussian_filter(height, sigma=3)
height_norm = (height - height.min()) / (height.max() - height.min())
height_uint16 = (height_norm * 65535).astype(np.uint16)
Image.fromarray(height_uint16, mode='I;16').save("mountain_heightmap.png")

