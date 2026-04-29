import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.util import random_noise


IMAGE_1_URL = "https://wtamu.edu/~cbaird/sq/images/refraction1.jpg"
IMAGE_3_URL = "https://www.fcbarcelona.com/fcbarcelona/photo/2025/12/24/494289a4-d698-4e48-aa41-5e424a337cca/WhatsApp-Image-2025-12-24-at-12.10.17.jpeg"


def load_image_from_url(url, max_size=(700, 700)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img.thumbnail(max_size)
    return np.array(img)


def to_grayscale(img):
    return rgb2gray(img)


def add_noise(img, amount=0.08):
    noisy = random_noise(img, mode="s&p", amount=amount)
    return noisy


def roberts_operator(gray_img, threshold=0.12):

    gray = gray_img.astype(float)
    h, w = gray.shape

    gradient = np.zeros((h - 1, w - 1))

    for y in range(h - 1):
        for x in range(w - 1):
            a = gray[y, x]
            b = gray[y, x + 1]
            c = gray[y + 1, x]
            d = gray[y + 1, x + 1]

            gx = a - d
            gy = b - c

            gradient[y, x] = np.sqrt(gx**2 + gy**2)

    gradient = gradient / gradient.max() if gradient.max() != 0 else gradient

    binary_edges = gradient > threshold

    return gradient, binary_edges




img1 = load_image_from_url(IMAGE_1_URL)
img3 = load_image_from_url(IMAGE_3_URL)

gray1 = to_grayscale(img1)
gray3 = to_grayscale(img3)

noisy_gray1 = add_noise(gray1, amount=0.05)

canny_sigma = 1.4
canny_low = 0.10
canny_high = 0.30

roberts_threshold = 0.12


canny1 = canny(gray1, sigma=canny_sigma, low_threshold=canny_low, high_threshold=canny_high)
canny_noisy = canny(noisy_gray1, sigma=canny_sigma, low_threshold=canny_low, high_threshold=canny_high)
canny3 = canny(gray3, sigma=canny_sigma, low_threshold=canny_low, high_threshold=canny_high)


roberts_grad1, roberts1 = roberts_operator(gray1, threshold=roberts_threshold)
roberts_grad_noisy, roberts_noisy = roberts_operator(noisy_gray1, threshold=roberts_threshold)
roberts_grad3, roberts3 = roberts_operator(gray3, threshold=roberts_threshold)





fig, axes = plt.subplots(3, 4, figsize=(18, 13))

images = [
    [gray1, canny1, roberts_grad1, roberts1],
    [noisy_gray1, canny_noisy, roberts_grad_noisy, roberts_noisy],
    [gray3, canny3, roberts_grad3, roberts3]
]

row_titles = [
    "Skaidrs objekts",
    "Attēls ar troksni",
    "Brīvi izvēlēts attēls"
]

col_titles = [
    "Oriģināls",
    "Canny",
    "Robertsa gradienta karte",
    "Robertsa binārā maska"
]

for row in range(3):
    for col in range(4):
        axes[row, col].imshow(images[row][col], cmap="gray")
        axes[row, col].axis("off")

        if row == 0:
            axes[row, col].set_title(col_titles[col], fontsize=12)

        if col == 0:
            axes[row, col].set_ylabel(row_titles[row], fontsize=12)

plt.tight_layout()
plt.show()
