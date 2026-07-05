import cv2
import numpy as np
import matplotlib.pyplot as plt

def niblack(image, r, k):
    prefix_image_brightness = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype=np.float64)
    prefix_image_square = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            prefix_image_brightness[i + 1, j + 1] = image[i, j] + prefix_image_brightness[i + 1, j] + prefix_image_brightness[i, j + 1] - prefix_image_brightness[i, j]
            prefix_image_square[i + 1, j + 1] = image[i, j] ** 2 + prefix_image_square[i + 1, j] + prefix_image_square[i, j + 1] - prefix_image_square[i, j]

    new_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for i in range (image.shape[0]):
        for j in range (image.shape[1]):

            top = i - r if (i - r) > 0 else 0
            bottom = i + r + 1 if (i + r + 1) < image.shape[0] + 1 else image.shape[0]
            
            left = j - r if (j - r) > 0 else 0
            right = j + r + 1 if (j + r + 1) < image.shape[1] + 1 else image.shape[1]

            square = (right - left) * (bottom - top)

            mid_brightness = (prefix_image_brightness[bottom, right] - prefix_image_brightness[top, right] - prefix_image_brightness[bottom, left] + prefix_image_brightness[top, left]) / square
            
            square_sum = (prefix_image_square[bottom, right] - prefix_image_square[top, right] - prefix_image_square[bottom, left] + prefix_image_square[top, left]) / square
            
            sigma = np.sqrt(max(square_sum - mid_brightness ** 2, 0))

            new_image[i, j] = 255 if (mid_brightness + k * sigma) < image[i, j] else 0

    return new_image


def multi_niblack(image, r, k):
    res_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    image_new = niblack(image, r, k)

    scale = 0.5
    d_image = cv2.resize(image, None, fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
    d_image_new = niblack(d_image, int(r * 0.5), k)

    scale = 0.25
    s_image = cv2.resize(image, None, fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
    s_image_new = niblack(s_image, int(r * 0.25), k)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            p1 = image_new[i, j]
            p2 = min(d_image_new[i // 2, j // 2], d_image_new[d_image_new.shape[0] - 1, d_image_new.shape[1]])
            p2 = min(s_image_new[i // 4, j // 4], s_image_new[d_image_new.shape[0] - 1, s_image_new.shape[1]])
            pixel_sum = image_new[i, j] + d_image_new[i // 2, j // 2] + s_image_new[i // 4, j // 4]
            res_image[i, j] = 255 if pixel_sum >= 382 else 0 

    return res_image
            

image_path = '0.jpg'
output_prefix = 'result'

image = cv2.imread(image_path)

if len(image.shape) == 3:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray_image = image

r = 15 
k = -0.2 

binary_image = multi_niblack(gray_image, r, k)

cv2.imwrite(f'{output_prefix}_binary.jpg', binary_image)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image, cmap='gray')
axes[0].set_title('init')
axes[0].axis('off')

axes[1].imshow(binary_image, cmap='gray')
axes[1].set_title(f'niblack (r={r}, k={k})')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(f'{output_prefix}_comparison.png', dpi=150, bbox_inches='tight')
plt.close()