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

def log_correction(img):
    img_f = img.astype(np.float32)
    c = 255 / np.log(1 + 255)
    result = c * np.log(1 + img_f)
    return np.clip(result, 0, 255).astype(np.uint8)

def histogram_equalization(img):
    result = np.zeros_like(img)

    for c in range(3):  # RGB kanāli
        channel = img[:, :, c]
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_masked = np.ma.masked_equal(cdf, 0)

        cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
        cdf_final = np.ma.filled(cdf_masked, 0).astype(np.uint8)

        result[:, :, c] = cdf_final[channel]

    return result

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

def show_results(img, title):
    img_log = log_correction(img)
    img_hist = histogram_equalization(img)

    fig = plt.figure(figsize=(15, 5))
    axs = [fig.add_subplot(1, 3, i+1) for i in range(3)]

    axs[0].imshow(img)
    axs[0].set_title(f"{title} - Oriģināls")

    axs[1].imshow(img_log)
    axs[1].set_title(f"{title} - Logaritmiskā korekcija")

    axs[2].imshow(img_hist)
    axs[2].set_title(f"{title} - Histogrammas izlīdzināšana")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    plot_histogram(img, f"{title} - Histogramma pirms korekcijas")
    plot_histogram(img_log, f"{title} - Histogramma pēc logaritmiskās korekcijas")
    plot_histogram(img_hist, f"{title} - Histogramma pēc histogrammas izlīdzināšanas")

show_results(img_dark, "Pārtumšots attēls")
show_results(img_bright, "Pārgaismots attēls")
show_results(img_gray, "Pelēcīgs attēls")
