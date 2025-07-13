import os
import kaggle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random

from matplotlib.colors import hsv_to_rgb
import matplotlib.gridspec as gridspec


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


def print_rgb_histograms():
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


def plot_hue_histogram(hist_h, healthy_hist=None, title='', percentile_range=None):
    plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 0.3])

    # --- Histograma ---
    ax_hist = plt.subplot(gs[0])
    ax_hist.plot(
        hist_h,
        color='crimson' if healthy_hist is not None else 'forestgreen',
        label='Disease Hue' if healthy_hist is not None else 'Healthy Hue'
    )

    if percentile_range is not None:
        p25, p75 = percentile_range
        ax_hist.fill_between(
            range(180),
            p25,
            p75,
            color='crimson' if healthy_hist is not None else 'forestgreen',
            alpha=0.3,
            label='25–75% Range'
        )

    if healthy_hist is not None:
        ax_hist.plot(healthy_hist, color='forestgreen', linestyle='--', linewidth=2, label='Healthy Hue')

    ax_hist.set_xlim(0, 179)
    ax_hist.set_ylim(0, 0.1)
    ax_hist.set_title(title, fontsize=14)
    ax_hist.set_xlabel('Hue value (0–179)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.legend()
    ax_hist.grid(True, linestyle=':', alpha=0.5)

    # --- Barra de colores Hue ---
    hue_gradient = np.linspace(0, 1, 180).reshape(1, -1)
    hue_colormap = np.stack([hue_gradient, np.ones_like(hue_gradient), np.ones_like(hue_gradient)], axis=-1)
    rgb_colormap = hsv_to_rgb(hue_colormap)

    ax_colorbar = plt.subplot(gs[1])
    ax_colorbar.imshow(rgb_colormap, aspect='auto', extent=[0, 179, 0, 1])
    ax_colorbar.set_xticks([])
    ax_colorbar.set_yticks([])
    ax_colorbar.set_ylabel("Hue", rotation=0, labelpad=15, va='center')
    ax_colorbar.set_xlim(0, 179)

    plt.tight_layout()
    plt.show()


def print_hue_histograms():
    base_dir = '../data/plantvillage/plantvillage dataset/segmented'
    disease_to_files = get_files_by_disease(base_dir)

    # --- Healthy ---
    healthy_files = disease_to_files.get('healthy', [])
    healthy_hists = []
    healthy_pixel_counts = []

    for file in healthy_files:
        img_bgr = cv.imread(file)
        img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

        mask = np.any(img_bgr > 10, axis=2)
        h_values = img_hsv[..., 0][mask]
        healthy_pixel_counts.append(h_values.size)

        h_hist, _ = np.histogram(h_values, bins=180, range=(0, 180))
        h_hist = h_hist / np.sum(h_hist)
        healthy_hists.append(h_hist)

    healthy_hists = np.array(healthy_hists)
    hist_h_healthy = np.median(healthy_hists, axis=0)
    healthy_p25 = np.percentile(healthy_hists, 25, axis=0)
    healthy_p75 = np.percentile(healthy_hists, 75, axis=0)

    avg_healthy_pixels = np.mean(healthy_pixel_counts)
    print(f"[Healthy] Avg. pixels used per image: {avg_healthy_pixels:.0f}")

    plot_hue_histogram(
        hist_h_healthy,
        title='Healthy - Median Hue Histogram',
        percentile_range=(healthy_p25, healthy_p75)
    )

    # --- Enfermedades ---
    for disease, files in disease_to_files.items():
        if disease == 'healthy':
            continue

        disease_hists = []
        disease_pixel_counts = []

        for file in files:
            img_bgr = cv.imread(file)
            img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

            mask = np.any(img_bgr > 10, axis=2)
            h_values = img_hsv[..., 0][mask]
            disease_pixel_counts.append(h_values.size)

            h_hist, _ = np.histogram(h_values, bins=180, range=(0, 180))
            h_hist = h_hist / np.sum(h_hist)
            disease_hists.append(h_hist)

        disease_hists = np.array(disease_hists)
        hist_h_disease = np.median(disease_hists, axis=0)
        hist_p25 = np.percentile(disease_hists, 25, axis=0)
        hist_p75 = np.percentile(disease_hists, 75, axis=0)

        avg_disease_pixels = np.mean(disease_pixel_counts)
        print(f"[{disease}] Avg. pixels used per image: {avg_disease_pixels:.0f}")

        plot_hue_histogram(
            hist_h_disease,
            healthy_hist=hist_h_healthy,
            title=f'{disease} - Median Hue Histogram vs Healthy',
            percentile_range=(hist_p25, hist_p75)
        )

