import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests
from PIL import Image
from io import BytesIO


IMAGE_URL_1 = "https://www.static.tu.berlin/fileadmin/www/_processed_/7/6/csm_Thumbnail_Rasen_935247d8c9.jpg"
IMAGE_URL_2 = "https://media.architecturaldigest.com/photos/66a914f1a958d12e0cc94a8e/16:9/w_2560%2Cc_limit/DSC_5903.jpg"
IMAGE_URL_3 = "https://cdn.mos.cms.futurecdn.net/ubcLwYk8iEPrGv3HtYpgnJ.jpg"
IMAGE_URL_4 = "https://img.runningwarehouse.com/watermark/rsg.php?path=/content_images/landing-pages/Running_Shoe_Components_2024/3fb43582-c9e9-45f3-8048-b96e5ad2df8b.jpeg&nw=780"
IMAGE_URL_5 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/960px-Cat_November_2010-1a.jpg"


def load_image_from_url(url, max_size=(320, 320)):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img.thumbnail(max_size)
    return np.array(img)


img1 = load_image_from_url(IMAGE_URL_1)
img2 = load_image_from_url(IMAGE_URL_2)
img3 = load_image_from_url(IMAGE_URL_3)
img4 = load_image_from_url(IMAGE_URL_4)
img5 = load_image_from_url(IMAGE_URL_5)

images = [img1, img2, img3, img4, img5]

row_titles = [
    "Zāles tekstūra",
    "Dzīvnieks",
    "Objekts",
    "Priekšmets",
    "Dabas attēls"
]


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def segment_with_median_canny_and_morphology(img, median_kernel_size=5, low_threshold=50, high_threshold=150, morph_kernel_size=5, morph_iterations=1):
    gray = to_gray(img)

    median_filtered = cv2.medianBlur(
        gray,
        median_kernel_size
    )

    canny_without_filter = cv2.Canny(
        gray,
        low_threshold,
        high_threshold
    )

    canny_with_median = cv2.Canny(
        median_filtered,
        low_threshold,
        high_threshold
    )

    kernel = np.ones(
        (morph_kernel_size, morph_kernel_size),
        np.uint8
    )

    closed = cv2.morphologyEx(
        canny_with_median,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=morph_iterations
    )

    return gray, median_filtered, canny_without_filter, canny_with_median, closed


MEDIAN_KERNEL_SIZE = 5
LOW_THRESHOLD = 50
HIGH_THRESHOLD = 150
MORPH_KERNEL_SIZE = 5
MORPH_ITERATIONS = 1


gray_images = []
median_filtered_images = []
canny_without_filter_results = []
canny_with_median_results = []
morphology_results = []

for img in images:
    gray, median_filtered, canny_without_filter, canny_with_median, closed = segment_with_median_canny_and_morphology(
        img,
        median_kernel_size=MEDIAN_KERNEL_SIZE,
        low_threshold=LOW_THRESHOLD,
        high_threshold=HIGH_THRESHOLD,
        morph_kernel_size=MORPH_KERNEL_SIZE,
        morph_iterations=MORPH_ITERATIONS
    )

    gray_images.append(gray)
    median_filtered_images.append(median_filtered)
    canny_without_filter_results.append(canny_without_filter)
    canny_with_median_results.append(canny_with_median)
    morphology_results.append(closed)



col_titles = [
    "Oriģināls",
    "Pelēktoņu attēls",
    "Mediānas filtrs",
    "Tikai Canny",
    "Mediānas filtrs + Canny",
    "Mediānas filtrs + Canny + Closing"
]

for row in range(5):
    fig, axes = plt.subplots(1, 6, figsize=(18, 4))

    row_images = [
        images[row],
        gray_images[row],
        median_filtered_images[row],
        canny_without_filter_results[row],
        canny_with_median_results[row],
        morphology_results[row]
    ]

    for col in range(6):
        if col == 0:
            axes[col].imshow(row_images[col])
        else:
            axes[col].imshow(row_images[col], cmap="gray")

        axes[col].axis("off")
        axes[col].set_title(col_titles[col], fontsize=10)

    fig.suptitle(row_titles[row], fontsize=14)
    plt.tight_layout()
    plt.show()


print("Izmantotie parametri:")
print("Median kernel size:", MEDIAN_KERNEL_SIZE)
print("Canny low threshold:", LOW_THRESHOLD)
print("Canny high threshold:", HIGH_THRESHOLD)
print("Morphology kernel size:", MORPH_KERNEL_SIZE)
print("Morphology iterations:", MORPH_ITERATIONS)
