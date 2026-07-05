import cv2
import numpy as np
import matplotlib.pyplot as plt


color_image = cv2.imread('table.jpg')
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), sigmaX = 1.5)

edge_image = cv2.Canny(blurred_image, threshold1 = 50, threshold2 = 150)

#----------------------------------------------------------------------------
# ручная реализация алгоритма функции houghlines 
# ys, xs = np.where(edge_image != 0)
# points = list(zip(xs, ys))

# height, width = edge_image.shape

# rho_max = int(np.sqrt(height ** 2 + width ** 2))

# theta_step = 1
# thetas_deg = np.arange(0, 180, theta_step)
# thetas_rad = np.deg2rad(thetas_deg)

# rho_step = 1
# rhos = np.arange(-rho_max, rho_max + 1, rho_step)

# accum = np.zeros((len(rhos), len(thetas_rad)), dtype = np.uint64)

# for (x, y) in points:
#     for theta_ind, theta_rad in enumerate(thetas_rad):
#         rho = x * np.cos(theta_rad) + y * np.sin(theta_rad)
        
#         rho_ind = int(np.round(rho / rho_step)) + rho_max

#         if 0 <= rho_ind < accum.shape[0]:
#             accum[rho_ind, theta_ind] += 1

# plt.figure(figsize=(8, 6))
# plt.imshow(accum, cmap='hot', aspect='auto', extent=[0, 180, -rho_max, rho_max])
# plt.xlabel('theta')
# plt.ylabel('rho')
# plt.title('accum')
# plt.colorbar(label='votes')
# plt.show()

# mode = np.max(accum)
# theshold = mode / 2

# maximums = np.where(accum >= theshold)
# rho_max_inds, theta_max_inds = maximums


lines = cv2.HoughLines(edge_image, rho = 1, theta = np.pi / 180, threshold = 100)
lines = lines.reshape(-1, 2)

angles_deg = []
for rho, theta in lines:
    angle = np.rad2deg(theta)

    if angle > 90:
        angle -= 180

    if abs(angle) > 40:    # в силу того, что я знаю, что угол не такой большой
        continue

    angles_deg.append(angle)

hist, bins = np.histogram(angles_deg, bins = 180, range = (-90, 90))
dom_angle = bins[np.argmax(hist)]

plt.figure(figsize=(12, 5))
plt.bar(bins[:-1], hist, width=1.0, edgecolor='black')
plt.xlabel('angle')
plt.ylabel('lines')
plt.title('histo')
plt.grid(True, alpha=0.3)

dominant_idx = np.argmax(hist)
plt.axvline(x=dom_angle, color='red', linestyle='--', label=f'dom angle: {dom_angle:.1f}°')
plt.legend()
plt.savefig('histogram.png', dpi=150, bbox_inches='tight')
plt.close()

#-----------------------------------------------------------------------------

height, width = gray_image.shape
center = (height // 2, width // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, dom_angle, scale = 1.0)

cos = np.abs(rotation_matrix[0, 0])
sin = np.abs(rotation_matrix[0, 1])
new_height = int(height * cos + width * sin)
new_width = int(height * sin + width * cos)
rotation_matrix[0, 2] += (new_width / 2) - center[0]
rotation_matrix[1, 2] += (new_height / 2) - center[1]

rotated_image = cv2.warpAffine(gray_image, rotation_matrix, (new_width, new_height), borderMode = cv2.BORDER_CONSTANT, borderValue = (0, 0, 0)) 

#---------------------------------------------------------------------

horisont_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

horisont_lines = cv2.morphologyEx(rotated_image, cv2.MORPH_OPEN, horisont_kernel)
vert_lines = cv2.morphologyEx(rotated_image, cv2.MORPH_OPEN, vert_kernel)

table_lines = cv2.min(horisont_lines, vert_lines)

result_image = cv2.subtract(rotated_image, table_lines)

# _, mask = cv2.threshold(table_lines, 150, 255, cv2.THRESH_BINARY_INV)
# result_image = rotated_image.copy()
# result_image[mask > 0] = 200

# result_image = cv2.medianBlur(result_image, 3)

fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# axes[0].imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
# axes[0].set_title('init')
# axes[0].axis('off')

axes[0].imshow(gray_image, cmap = 'gray')
axes[0].set_title(f'gray')
axes[0].axis('off')

# axes[2].imshow(blurred_image, cmap = 'gray')
# axes[2].set_title(f'blur, gauss, (5, 5)')
# axes[2].axis('off')

# axes[3].imshow(edge_image, cmap = 'gray')
# axes[3].set_title(f'canny (threshold1 = 50, threshold2 = 150)')
# axes[3].axis('off')

axes[1].imshow(rotated_image, cmap = 'gray')
axes[1].set_title(f'rotated ( angle = {dom_angle})')
axes[1].axis('off')

axes[2].imshow(result_image, cmap = 'gray')
axes[2].set_title(f'result')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(f'result.png', dpi = 150, bbox_inches = 'tight')
plt.close()