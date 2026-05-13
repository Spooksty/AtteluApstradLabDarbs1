import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests
from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans

IMAGE_URL_1 = "https://img.freepik.com/premium-photo/many-different-models-printed-3d-printer-gray-black-objects-printed_507658-6884.jpg?w=360"
IMAGE_URL_2 = "https://hips.hearstapps.com/hmg-prod/images/ferrari-e-suv-2-copy-680287cac36b2.jpg?crop=1.00xw:0.838xh;0,0.0673xh"
IMAGE_URL_3 = "https://assets.goal.com/images/v3/blt411f83ea5a5aca9d/06.gif?auto=webp&format=pjpg&width=3840&quality=60"


def load_image_from_url(url, resize_to=(400, 400)):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize(resize_to, Image.LANCZOS)
    return np.array(img)


img1 = load_image_from_url(IMAGE_URL_1)
img2 = load_image_from_url(IMAGE_URL_2)
img3 = load_image_from_url(IMAGE_URL_3)

images = [img1, img2, img3]

row_titles = [
    "Pelēktoņu attēls ar objektiem",
    "Cilvēks/dzīvnieks/auto",
    "Brīvi izvēlēts attēls"
]



def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussian_adaptive_threshold(img, block_size=35, C=5):
    gray = to_gray(img)

    result = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        C
    )

    return gray, result


def kmeans_segmentation(img, k=3):
  pixels = img.reshape((-1, 3))

  kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
  labels = kmeans.fit_predict(pixels)

  centers = kmeans.cluster_centers_.astype(np.uint8)
  segmented_pixels = centers[labels]

  segmented_img = segmented_pixels.reshape(img.shape)

  return segmented_img


BLOCK_SIZE = 35
C = 5
K = 8

gray_images = []
threshold_results = []
kmeans_results = []

for img in images:
    gray, thresholded = gaussian_adaptive_threshold(img, block_size=BLOCK_SIZE, C=C)
    kmeans_img = kmeans_segmentation(img, k=K)

    gray_images.append(gray)
    threshold_results.append(thresholded)
    kmeans_results.append(kmeans_img)




fig, axes = plt.subplots(3, 4, figsize=(18, 13))

col_titles = [
    "Oriģināls",
    "Pelēktoņu attēls",
    "Gausa adaptīvā sliekšņošana",
    "K-Means segmentācija"
]

for row in range(3):
    axes[row, 0].imshow(images[row])
    axes[row, 1].imshow(gray_images[row], cmap="gray")
    axes[row, 2].imshow(threshold_results[row], cmap="gray")
    axes[row, 3].imshow(kmeans_results[row])

    for col in range(4):
        axes[row, col].axis("off")

        if row == 0:
            axes[row, col].set_title(col_titles[col], fontsize=12)

        if col == 0:
            axes[row, col].set_ylabel(row_titles[row], fontsize=12)

plt.tight_layout()
plt.show()

