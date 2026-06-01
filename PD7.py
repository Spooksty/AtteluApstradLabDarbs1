import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image



BASE = "/content/drive/MyDrive/Colab Notebooks" 

image_files = [
    f"{BASE}/Abyssinian_1r (1).jpg",
    f"{BASE}/Abyssinian_1r (2).jpg",
    f"{BASE}/Abyssinian_1r (3).jpg"
]

mask_files = [
    f"{BASE}/Abyssinian_1.png",
    f"{BASE}/Abyssinian_2.png",
    f"{BASE}/Abyssinian_3.png"
]

names = [
    "Abyssinian 1",
    "Abyssinian 2",
    "Abyssinian 3"
]


def load_rgb(path, max_size=(360, 360)):
    img = Image.open(path).convert("RGB")
    img.thumbnail(max_size)
    return np.array(img)


def load_mask(path, target_shape):
    mask = Image.open(path).convert("L")
    mask = mask.resize((target_shape[1], target_shape[0]), Image.NEAREST)
    mask = np.array(mask)

    print(path, "unikālās vērtības:", np.unique(mask))


    binary_mask = np.where((mask == 1) | (mask == 3), 255, 0)

    return binary_mask.astype(np.uint8)


images = []
gt_object_masks = []

for img_path, mask_path in zip(image_files, mask_files):
    img = load_rgb(img_path)
    mask = load_mask(mask_path, img.shape[:2])
    images.append(img)
    gt_object_masks.append(mask)



def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def segment_with_median_and_canny(
    img,
    median_kernel_size=5,
    low_threshold=80,
    high_threshold=180
):
    gray = to_gray(img)

    median_filtered = cv2.medianBlur(
        gray,
        median_kernel_size
    )

    canny_edges = cv2.Canny(
        median_filtered,
        low_threshold,
        high_threshold
    )

    return gray, median_filtered, canny_edges


def dilate_edges(edge_mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(edge_mask, kernel, iterations=1)

def ground_truth_edges_from_mask(mask):
    return cv2.Canny(mask, 50, 150)



def calculate_iou(gt_mask, pred_mask):
    gt = gt_mask > 0
    pred = pred_mask > 0

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    if union == 0:
        return 0

    return intersection / union


def calculate_dice(gt_mask, pred_mask):
    gt = gt_mask > 0
    pred = pred_mask > 0

    intersection = np.logical_and(gt, pred).sum()
    total = gt.sum() + pred.sum()

    if total == 0:
        return 0

    return 2 * intersection / total
    return 2 * intersection / total if total != 0 else 0



MEDIAN_KERNEL_SIZE = 5
LOW_THRESHOLD = 80
HIGH_THRESHOLD = 180

results = []

for img, gt_object_mask, name in zip(images, gt_object_masks, names):
    gray, median_filtered, predicted_edges = segment_with_median_and_canny(
        img,
        median_kernel_size=MEDIAN_KERNEL_SIZE,
        low_threshold=LOW_THRESHOLD,
        high_threshold=HIGH_THRESHOLD
    )

    gt_edges = ground_truth_edges_from_mask(gt_object_mask)

    gt_edges_dilated = dilate_edges(gt_edges, kernel_size=5)
    predicted_edges_dilated = dilate_edges(predicted_edges, kernel_size=5)

    iou_dilated = calculate_iou(gt_edges_dilated, predicted_edges_dilated)
    dice_dilated = calculate_dice(gt_edges_dilated, predicted_edges_dilated)

    iou = calculate_iou(gt_edges, predicted_edges)
    dice = calculate_dice(gt_edges, predicted_edges)

    results.append({
        "Attēls": name,
        "IoU": round(iou, 4),
        "Dice": round(dice, 4)
    })

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    axes[0].imshow(img)
    axes[0].set_title("Oriģināls")

    axes[1].imshow(gt_object_mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Īstā objekta maska")

    axes[2].imshow(gt_edges, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Ground truth malas")

    axes[3].imshow(predicted_edges, cmap="gray", vmin=0, vmax=255)
    axes[3].set_title(f"Canny malas\nIoU={iou:.3f}, Dice={dice:.3f}")

    axes[4].imshow(predicted_edges_dilated, cmap="gray", vmin=0, vmax=255)
    axes[4].set_title(f"Ar dilation\nIoU={iou_dilated:.3f}\nDice={dice_dilated:.3f}")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(name, fontsize=14)
    plt.tight_layout()
    plt.show()

df_results = pd.DataFrame(results)
df_results
