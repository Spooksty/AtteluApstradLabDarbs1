
IMAGE_URL_1 = "https://cdn.britannica.com/16/277116-050-AE9CDB07/Lamine-Yamal-Scores-For-FC-Barcelona-In-LaLiga-2025-Clash-Against-Rayo-Vallecano-Madrid.jpg"
IMAGE_URL_2 = "https://assets.goal.com/images/v3/blt6d05eb02265c690f/GOAL_-_Blank_WEB_-_Facebook_-_2023-09-26T201135.941.png?auto=webp&format=pjpg&width=3840&quality=60"

from IPython.display import display, HTML
display(HTML("""
<style>
  div.output_area, div.output_subarea, figure, img, .output_png {
    page-break-inside: avoid !important;
    break-inside: avoid !important;
  }
</style>
"""))

import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests
from PIL import Image
from io import BytesIO
%matplotlib inline

def load_image_from_url(url, resize_to=(400, 400)):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=15)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize(resize_to, Image.LANCZOS)
    return np.array(img)

img1 = load_image_from_url(IMAGE_URL_1)
img2 = load_image_from_url(IMAGE_URL_2)



fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Oriģinālie attēli', fontsize=15, fontweight='bold')
axes[0].imshow(img1); axes[0].set_title('Attēls 1'); axes[0].axis('off')
axes[1].imshow(img2); axes[1].set_title('Attēls 2'); axes[1].axis('off')
plt.tight_layout()
plt.savefig('01_originalie.png', dpi=150, bbox_inches='tight')
plt.show()



def add_salt_and_pepper_noise(image, noise_prob=0.05, seed=42):
    np.random.seed(seed)
    noisy = image.copy().astype(np.float64)
    random_matrix = np.random.random(image.shape[:2])
    noisy[random_matrix < (noise_prob / 2)] = 255.0
    noisy[(random_matrix >= (noise_prob / 2)) & (random_matrix < noise_prob)] = 0.0
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_gaussian_noise(image, mean=0, sigma=25, seed=42):
    np.random.seed(seed)
    noisy = image.astype(np.float64) + np.random.normal(mean, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)

SALT_PEPPER_PROB = 0.05
GAUSSIAN_MEAN = 0
GAUSSIAN_SIGMA = 25

img1_sp    = add_salt_and_pepper_noise(img1, noise_prob=SALT_PEPPER_PROB)
img2_sp    = add_salt_and_pepper_noise(img2, noise_prob=SALT_PEPPER_PROB)
img1_gauss = add_gaussian_noise(img1, mean=GAUSSIAN_MEAN, sigma=GAUSSIAN_SIGMA)
img2_gauss = add_gaussian_noise(img2, mean=GAUSSIAN_MEAN, sigma=GAUSSIAN_SIGMA)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Sāls un piparu troksnis (noise_prob={SALT_PEPPER_PROB})', fontsize=15, fontweight='bold')
axes[0].imshow(img1_sp); axes[0].set_title('Attēls 1'); axes[0].axis('off')
axes[1].imshow(img2_sp); axes[1].set_title('Attēls 2'); axes[1].axis('off')
plt.tight_layout()
plt.savefig('02_sals_pipari.png', dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Gausa troksnis (mean={GAUSSIAN_MEAN}, sigma={GAUSSIAN_SIGMA})', fontsize=15, fontweight='bold')
axes[0].imshow(img1_gauss); axes[0].set_title('Attēls 1'); axes[0].axis('off')
axes[1].imshow(img2_gauss); axes[1].set_title('Attēls 2'); axes[1].axis('off')
plt.tight_layout()
plt.savefig('03_gausa_troksnis.png', dpi=150, bbox_inches='tight')
plt.show()


MEAN_KERNEL          = 3
MEDIAN_KERNEL        = 3
GAUSSIAN_KERNEL      = 3
GAUSSIAN_SIGMA_FILT  = 1.0

noisy = [img1_sp, img2_sp, img1_gauss, img2_gauss]
labels = [
    'Att.1 + Sāls&Pipari',
    'Att.2 + Sāls&Pipari',
    'Att.1 + Gausa',
    'Att.2 + Gausa',
]
originals = [img1, img2, img1, img2]

mean_res    = [cv2.blur(img, (MEAN_KERNEL, MEAN_KERNEL))                                          for img in noisy]
median_res  = [cv2.medianBlur(img, MEDIAN_KERNEL)                                                 for img in noisy]
gaussian_res= [cv2.GaussianBlur(img, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), GAUSSIAN_SIGMA_FILT)     for img in noisy]

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle(f'Vidējais filtrs (kernel {MEAN_KERNEL}×{MEAN_KERNEL})', fontsize=15, fontweight='bold')
for ax, img, lbl in zip(axes, mean_res, labels):
    ax.imshow(img); ax.set_title(lbl, fontsize=10); ax.axis('off')
plt.tight_layout()
plt.savefig('04_videjais_filtrs.png', dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle(f'Mediānas filtrs (kernel {MEDIAN_KERNEL}×{MEDIAN_KERNEL})', fontsize=15, fontweight='bold')
for ax, img, lbl in zip(axes, median_res, labels):
    ax.imshow(img); ax.set_title(lbl, fontsize=10); ax.axis('off')
plt.tight_layout()
plt.savefig('05_medianas_filtrs.png', dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle(f'Gausa filtrs (kernel {GAUSSIAN_KERNEL}×{GAUSSIAN_KERNEL}, σ={GAUSSIAN_SIGMA_FILT})', fontsize=15, fontweight='bold')
for ax, img, lbl in zip(axes, gaussian_res, labels):
    ax.imshow(img); ax.set_title(lbl, fontsize=10); ax.axis('off')
plt.tight_layout()
plt.savefig('06_gausa_filtrs.png', dpi=150, bbox_inches='tight')
plt.show()
