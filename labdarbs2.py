import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

url_dark = "https://thelenslounge.com/wp-content/uploads/2024/08/low-key-light-portrait-1.jpg"
url_bright = "https://live.staticflickr.com/1115/5113330606_30bd4a5235_b.jpg"
url_gray = "https://my.willamette.edu/site/digital-accessibility/images/colorcontrast-example2.jpg"

def load_rgb_from_url(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)

img_dark = load_rgb_from_url(url_dark)
img_bright = load_rgb_from_url(url_bright)
img_gray = load_rgb_from_url(url_gray)

def gamma_correction(img, gamma):
    img_f = img.astype(np.float32) / 255.0
    result = np.power(img_f, gamma)
    return np.clip(result * 255, 0, 255).astype(np.uint8)

def linear_contrast_stretch(img):
    result = np.zeros_like(img)

    for c in range(3):  # RGB kanāli
        channel = img[:, :, c].astype(np.float32)
        c_min = channel.min()
        c_max = channel.max()

        if c_max > c_min:
            stretched = (channel - c_min) / (c_max - c_min) * 255
        else:
            stretched = channel

        result[:, :, c] = np.clip(stretched, 0, 255)

    return result.astype(np.uint8)

def plot_histogram(img, title):
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(6,4))

    for i, color in enumerate(colors):
        hist, bins = np.histogram(img[:, :, i].flatten(), bins=256, range=(0, 256))
        plt.plot(hist, color=color)

    plt.title(title)
    plt.xlim([0, 255])
    plt.xlabel("Spilgtuma vērtība")
    plt.ylabel("Pikseļu skaits")
    plt.show()

def show_results(img, title, gamma_value):
    img_gamma = gamma_correction(img, gamma_value)
    img_linear = linear_contrast_stretch(img)

    fig = plt.figure(figsize=(15, 5))
    axs = [fig.add_subplot(1, 3, i+1) for i in range(3)]

    axs[0].imshow(img)
    axs[0].set_title(f"{title} - Oriģināls")

    axs[1].imshow(img_gamma)
    axs[1].set_title(f"{title} - Gamma korekcija")

    axs[2].imshow(img_linear)
    axs[2].set_title(f"{title} - Lineārā pārveidošana")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    plot_histogram(img, f"{title} - Histogramma pirms korekcijas")
    plot_histogram(img_gamma, f"{title} - Histogramma pēc gamma korekcijas")
    plot_histogram(img_linear, f"{title} - Histogramma pēc lineārās pārveidošanas")

show_results(img_dark, "Pārtumšots attēls", gamma_value=0.6)
show_results(img_bright, "Pārgaismots attēls", gamma_value=1.5)
show_results(img_gray, "Pelēcīgs attēls", gamma_value=0.8)
