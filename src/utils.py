import os
import kaggle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random

def download_plantvillage_dataset(download_path='data/plantvillage'):
    """
    Downloads the PlantVillage dataset from Kaggle to the specified path.
    Requires Kaggle API credentials.
    """
    os.makedirs(download_path, exist_ok=True)
    kaggle.api.dataset_download_files(
        'abdallahalidev/plantvillage-dataset',
        path=download_path,
        unzip=True
    )
    print(f'Dataset downloaded and extracted to {download_path}')


def get_files_by_disease(type_dir, prefix='Tomato___'):
    dirs = [d for d in os.listdir(type_dir) if d.startswith(prefix)]
    disease_to_files = {}
    for folder in dirs:
        disease = folder.split('___', 1)[1]
        folder_path = os.path.join(type_dir, folder)
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                 f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        disease_to_files[disease] = files
    return disease_to_files


def plot_healthy_rgb_histogram(example_files, hist_r, hist_g, hist_b):
    # Plot examples and histogram
    plt.figure(figsize=(16, 6))
    for idx, file in enumerate(example_files):
        img_bgr = cv.imread(file)
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        plt.subplot(2, 4, idx + 1)
        plt.imshow(img_rgb)
        plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.plot(hist_r, color='r', label='Red')
    plt.plot(hist_g, color='g', label='Green')
    plt.plot(hist_b, color='b', label='Blue')
    plt.title(f'Average RGB Histogram: Tomato healthy')
    plt.xlabel('Pixel value')
    plt.ylabel('Average Frequency')
    plt.ylim(0, 800)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_disease_rgb_histogram(example_files, hist_r, hist_g, hist_b, healthy_hist, disease):
    plt.figure(figsize=(16, 6))
    for idx, file in enumerate(example_files):
        img_bgr = cv.imread(file)
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        plt.subplot(2, 4, idx + 1)
        plt.imshow(img_rgb)
        plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.plot(hist_r, color='r', label='Red (disease)')
    plt.plot(hist_g, color='g', label='Green (disease)')
    plt.plot(hist_b, color='b', label='Blue (disease)')
    # Add healthy as dashed
    hr, hg, hb = healthy_hist
    plt.plot(hr, color='r', linestyle='--', label='Red (healthy)')
    plt.plot(hg, color='g', linestyle='--', label='Green (healthy)')
    plt.plot(hb, color='b', linestyle='--', label='Blue (healthy)')
    plt.title(f'Average RGB Histogram: Tomato {disease} vs healthy')
    plt.xlabel('Pixel value')
    plt.ylabel('Average Frequency')
    plt.ylim(0, 800)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_histograms():
    base_dir = '../data/plantvillage/plantvillage dataset'
    types = ['color', 'segmented']

    # Get files for both color and segmented
    disease_to_files = {t: get_files_by_disease(os.path.join(base_dir, t)) for t in types}

    # --- 1. Show healthy histograms and examples for color and segmented ---
    healthy_hist = {}
    healthy_examples = {}

    for t in types:
        files = disease_to_files[t].get('healthy', [])
        n_examples = min(4, len(files))
        example_files = random.sample(files, n_examples)
        healthy_examples[t] = example_files

        hist_r = np.zeros(256)
        hist_g = np.zeros(256)
        hist_b = np.zeros(256)
        n_images = 0

        for file in files:
            img_bgr = cv.imread(file)
            img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
            if t == 'segmented':
                threshold = 40
                mask = np.any(img_rgb > threshold, axis=-1)
            else:
                mask = np.ones(img_rgb.shape[:2], dtype=bool)
            for i, hist in enumerate([hist_r, hist_g, hist_b]):
                channel = img_rgb[..., i][mask]
                h, _ = np.histogram(channel, bins=256, range=(0, 256))
                hist += h
            n_images += 1

        if n_images > 0:
            hist_r /= n_images
            hist_g /= n_images
            hist_b /= n_images

        healthy_hist[t] = (hist_r, hist_g, hist_b)

        # Plot examples and histogram
        plot_healthy_rgb_histogram(example_files, hist_r, hist_g, hist_b)

    # --- 2. For each disease, show color and segmented histograms with healthy as dashed, and 4 example images each ---
    for disease in disease_to_files['color']:
        if disease == 'healthy':
            continue
        for t in types:
            files = disease_to_files[t].get(disease, [])
            n_examples = min(4, len(files))
            example_files = random.sample(files, n_examples)

            hist_r = np.zeros(256)
            hist_g = np.zeros(256)
            hist_b = np.zeros(256)
            n_images = 0

            for file in files:
                img_bgr = cv.imread(file)
                img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
                if t == 'segmented':
                    threshold = 40
                    mask = np.any(img_rgb > threshold, axis=-1)
                else:
                    mask = np.ones(img_rgb.shape[:2], dtype=bool)
                for i, hist in enumerate([hist_r, hist_g, hist_b]):
                    channel = img_rgb[..., i][mask]
                    h, _ = np.histogram(channel, bins=256, range=(0, 256))
                    hist += h
                n_images += 1

            if n_images > 0:
                hist_r /= n_images
                hist_g /= n_images
                hist_b /= n_images

            # Plot examples and histogram
            plot_disease_rgb_histogram(
                example_files, hist_r, hist_g, hist_b,
                healthy_hist[t], disease
            )