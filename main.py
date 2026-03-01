import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
# https://media.cnn.com/api/v1/images/stellar/prod/gettyimages-2234200789.jpg?c=original ronaldo
# https://cdn.britannica.com/35/238335-050-2CB2EB8A/Lionel-Messi-Argentina-Netherlands-World-Cup-Qatar-2022.jpg messi

url_a = "https://d7hftxdivxxvm.cloudfront.net/?quality=80&resize_to=width&src=https%3A%2F%2Fartsy-media-uploads.s3.amazonaws.com%2F1E2blfmrxuSPUH-2oc08gw%252F14273043642_d63ded6c05_o%2B%25281%2529.png&width=910"
url_b = "https://thumbs.dreamstime.com/b/landscape-sky-full-moon-seascape-to-night-serenity-nature-beautiful-view-sea-many-stars-attractive-dark-cloud-141554532.jpg"

def load_rgb_from_url(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    return img

imgA = load_rgb_from_url(url_a)
imgB = load_rgb_from_url(url_b).resize(imgA.size, Image.BILINEAR)

A = np.asarray(imgA, dtype=np.float32) / 255.0
B = np.asarray(imgB, dtype=np.float32) / 255.0

def clip01(x):
    return np.clip(x, 0.0, 1.0)

C_sub = clip01(A + B - 1.0)
C_mul = clip01(A * B)

eps = 1e-6
C_dodge = clip01(B / np.maximum(1.0 - A, eps))

d = 0.6
C_opacity = clip01(d * A + (1.0 - d) * B)

to_u8 = lambda x: (clip01(x) * 255.0 + 0.5).astype(np.uint8)

fig = plt.figure(figsize=(12, 8))
axs = [fig.add_subplot(2, 3, i+1) for i in range(6)]

axs[0].imshow(to_u8(A)); axs[0].set_title("A (Original)")
axs[1].imshow(to_u8(B)); axs[1].set_title("B (Original)")
axs[2].imshow(to_u8(C_sub)); axs[2].set_title("Subtraction / Linear Burn: A + B - 1")
axs[3].imshow(to_u8(C_mul)); axs[3].set_title("Multiply: A * B")
axs[4].imshow(to_u8(C_dodge)); axs[4].set_title("Dodge / Color Dodge: B / (1 - A)")
axs[5].imshow(to_u8(C_opacity)); axs[5].set_title(f"Opacity: d*A + (1-d)*B (d={d})")

for ax in axs:
    ax.axis("off")

plt.tight_layout()
plt.show()
