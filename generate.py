import numpy as np
from PIL import Image
from scipy.spatial import Voronoi
from scipy.ndimage import distance_transform_edt, gaussian_filter
import noise
from skimage.draw import line_aa

size = 1081
num_points = np.random.randint(10, 20)
points = np.random.rand(num_points, 2) * size
vor = Voronoi(points)
ridge_lines = []
for simplex in vor.ridge_vertices:
    if -1 not in simplex:
        p1 = vor.vertices[simplex[0]]
        p2 = vor.vertices[simplex[1]]
        ridge_lines.append((p1, p2))
if ridge_lines:
    ridge = ridge_lines[np.random.randint(0, len(ridge_lines))]
    mask = np.zeros((size, size), dtype=np.uint8)
    x0, y0 = int(ridge[0][0]), int(ridge[0][1])
    x1, y1 = int(ridge[1][0]), int(ridge[1][1])
    rr, cc, _ = line_aa(y0, x0, y1, x1)
    valid = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
    mask[rr[valid], cc[valid]] = 1
else:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size//2, size//2] = 1
dist = distance_transform_edt(mask)
dist = np.max(dist) - dist
noise_scale = np.random.uniform(50, 150)
noise_octaves = np.random.randint(4, 8)
noise_persistence = np.random.uniform(0.3, 0.7)
noise_lacunarity = np.random.uniform(1.5, 2.5)
noise_array = np.zeros((size, size), dtype=np.float32)
noise_base = np.random.randint(0, 100)
for i in range(size):
    for j in range(size):
        noise_array[i, j] = noise.pnoise2(i / noise_scale, j / noise_scale, octaves=noise_octaves, persistence=noise_persistence, lacunarity=noise_lacunarity, repeatx=size, repeaty=size, base=noise_base)
noise_array = (noise_array - noise_array.min()) / (noise_array.max() - noise_array.min())
height = dist + noise_array * np.max(dist) * np.random.uniform(0.1, 0.3)
height = gaussian_filter(height, sigma=np.random.uniform(1, 5))
height_norm = (height - height.min()) / (height.max() - height.min())
height_uint16 = (height_norm * 65535).astype(np.uint16)
Image.fromarray(height_uint16, mode='I;16').save("random_voronoi_mountain_heightmap.png")

