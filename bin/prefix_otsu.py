import cv2
import numpy as np
import matplotlib.pyplot as plt

def otsu(histogramm):
    prefix_list_brightness = [0] * 256
    prefix_list_pixel = [0] * 256

    for i in range(256):
        if i == 0:
            prefix_list_brightness[i] = histogramm[i] * i
            prefix_list_pixel[i] = histogramm[i]
        else:
            prefix_list_brightness[i] = prefix_list_brightness[i - 1] + histogramm[i] * i
            prefix_list_pixel[i] = prefix_list_pixel[i - 1] + histogramm[i]

    best_value = -1
    best_threshold = 0
    
    for threshold in range(256):
        if prefix_list_pixel[threshold] == 0 or (prefix_list_pixel[255] - prefix_list_pixel[threshold]) == 0:
            continue

        w_left = prefix_list_pixel[threshold] / prefix_list_pixel[255]
        mu_left = prefix_list_brightness[threshold] / prefix_list_pixel[threshold]

        w_right = (prefix_list_pixel[255] - prefix_list_pixel[threshold]) / prefix_list_pixel[255]
        mu_right = (prefix_list_brightness[255] - prefix_list_brightness[threshold]) / (prefix_list_pixel[255] - prefix_list_pixel[threshold])

        value = (mu_left - mu_right) ** 2 * w_right * w_left

        if value > best_value:
            best_value = value
            best_threshold = threshold
        
    return best_threshold


image_path = '1.jpg'
output_prefix = 'result' 

image = cv2.imread(image_path)

histogramm, _ = np.histogram(image, bins = 256, range = (0, 256))
histogramm = histogramm.tolist()

threshold = otsu(histogramm)

binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)

cv2.imwrite(f'{output_prefix}_binary.jpg', binary_image)
plt.close()

fig, axes = plt.subplots(1, 2, figsize = (12, 6))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('init')
axes[0].axis('off')

axes[1].imshow(binary_image, cmap = 'gray')
axes[1].set_title(f'bin (threshold = {threshold})')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(f'{output_prefix}_comparison.png', dpi = 150, bbox_inches = 'tight')
plt.close()