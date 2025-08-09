import os
import kaggle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
import pandas as pd

from matplotlib.colors import hsv_to_rgb
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import euclidean
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import seaborn as sns


def features_extract(df):
    def extract_features(path):
        try:
            img = cv.imread(path)
            img = img[:128, :128]
            features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True, channel_axis=-1)
            return features
        except Exception as e:
            print(f"Error con {path}: {e}")
            return None

    hog_features = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        feat = extract_features(row['File'])
        if feat is not None:
            hog_features.append(feat)
            valid_indices.append(idx)

    df_features = df.loc[valid_indices].reset_index(drop=True)
    hog_features = np.array(hog_features)

    return df_features, hog_features


def plt_distance_species_with_same_disease(features, hog):
    features["x"] = hog[:, 0]
    features["y"] = hog[:, 1]

    distancias_por_enfermedad = {}

    for enfermedad in features["Enfermedad"].unique():
        sub_df = features[features["Enfermedad"] == enfermedad]
        especies = sub_df["Especie"].unique()

        centroides = {}
        for especie in especies:
            puntos = sub_df[sub_df["Especie"] == especie][["x", "y"]].values
            centroides[especie] = puntos.mean(axis=0)

        distancias = []
        for i, esp1 in enumerate(especies):
            for esp2 in especies[i+1:]:
                d = euclidean(centroides[esp1], centroides[esp2])
                distancias.append(d)

        if distancias:
            distancias_por_enfermedad[enfermedad] = np.mean(distancias)

    for enfermedad in sorted(distancias_por_enfermedad, key=distancias_por_enfermedad.get):
        subset = features[features["Enfermedad"] == enfermedad]
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=subset, x="x", y="y", hue="Especie", s=80)
        plt.title(f"{enfermedad} (distancia media entre especies: {distancias_por_enfermedad[enfermedad]:.2f})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


def plt_distance_species_with_same_disease_pca(df):
    df_features, hog_features = features_extract(df)

    pca = PCA(n_components=2)
    hog_pca = pca.fit_transform(hog_features)

    plt_distance_species_with_same_disease(df_features, hog_pca)


def plt_distance_species_with_same_disease_tsne(df):
    df_features, hog_features = features_extract(df)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    hog_tsne = tsne.fit_transform(hog_features)

    plt_distance_species_with_same_disease(df_features, hog_tsne)


def plt_distance_healthy_species_pca(df):
    df_sano = df[df["Enfermedad"].str.lower() == "sano"].copy()
    df_features, hog_features = features_extract(df_sano)

    pca = PCA(n_components=2)
    hog_pca = pca.fit_transform(hog_features)

    plt_distance_healthy_species(df_features, hog_pca)


def plt_distance_healthy_species_tsne(df):
    df_sano = df[df["Enfermedad"].str.lower() == "sano"].copy()
    df_features, hog_features = features_extract(df_sano)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    hog_tsne = tsne.fit_transform(hog_features)

    plt_distance_healthy_species(df_features, hog_tsne)


def plt_distance_healthy_species(df_sano_valid, hog):

    df_sano_valid["x"] = hog[:, 0]
    df_sano_valid["y"] = hog[:, 1]

    especies_sanas = df_sano_valid["Especie"].unique()
    centroides = {}

    for especie in especies_sanas:
        puntos = df_sano_valid[df_sano_valid["Especie"] == especie][["x", "y"]].values
        if len(puntos) > 0:
            centroides[especie] = puntos.mean(axis=0)

    especies_ordenadas = sorted(centroides.keys())
    matriz_dist = pd.DataFrame(index=especies_ordenadas, columns=especies_ordenadas, dtype=float)

    for i, esp1 in enumerate(especies_ordenadas):
        for j, esp2 in enumerate(especies_ordenadas):
            if i <= j:
                d = euclidean(centroides[esp1], centroides[esp2])
                matriz_dist.loc[esp1, esp2] = d
                matriz_dist.loc[esp2, esp1] = d

    matriz_dist["Distancia Media"] = matriz_dist.apply(
        lambda row: row.dropna().drop(labels=row.name).mean(), axis=1
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_dist, annot=True, fmt=".2f", cmap="viridis", square=True)
    plt.title("Distancia euclidiana entre especies (solo imágenes sanas)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def build_dataset(base_dir):
    subfolders = ['color', 'grayscale', 'segmented']
    data = []

    for sub in subfolders:
        sub_path = os.path.join(base_dir, sub)
        if not os.path.exists(sub_path):
            continue
        for folder in os.listdir(sub_path):
            folder_path = os.path.join(sub_path, folder)
            if not os.path.isdir(folder_path):
                continue
            species, disease = folder.split('___', 1)
            if disease == 'healthy':
                healthy = True
                disease = None
            else:
                healthy = False
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    data.append({
                        'Format': sub,
                        'Species': species,
                        'Healthy': healthy,
                        'Disease': disease,
                        'Folder': folder_path,
                        'FileName': file,
                        'File': folder_path + '/' + file
                    })

    return pd.DataFrame(data)


def build_color_spa_dataset(base_dir):
    # Species mapping
    species_es_map = {
        'Strawberry': 'Fresa',
        'Grape': 'Uva',
        'Potato': 'Papa',
        'Blueberry': 'Arándano',
        'Corn_(maize)': 'Maíz',
        'Tomato': 'Tomate',
        'Peach': 'Durazno',
        'Pepper,_bell': 'Pimiento',
        'Orange': 'Naranja',
        'Cherry_(including_sour)': 'Cereza',
        'Apple': 'Manzana',
        'Raspberry': 'Frambuesa',
        'Squash': 'Calabaza',
        'Soybean': 'Soja'
    }

    # Disease mapping
    disease_es_map = {
        'Black_rot': 'Podredumbre negra',
        'Early_blight': 'Tizón temprano',
        'Target_Spot': 'Mancha diana',
        'Late_blight': 'Tizón tardío',
        'Tomato_mosaic_virus': 'Virus del mosaico',
        'Haunglongbing_(Citrus_greening)': 'Huanglongbing (HLB) / enverdecimiento de los cítricos',
        'Leaf_Mold': 'Moho de la hoja',
        'Leaf_blight_(Isariopsis_Leaf_Spot)': 'Tizón de la hoja (mancha foliar por Isariopsis)',
        'Powdery_mildew': 'Oídio (cenicilla o polvillo blanco)',
        'Cedar_apple_rust': 'Roya del manzano y cedro',
        'Bacterial_spot': 'Mancha bacteriana',
        'Common_rust_': 'Roya común',
        'Esca_(Black_Measles)': 'Esca (manchas negras)',
        'Tomato_Yellow_Leaf_Curl_Virus': 'Virus del rizado amarillo de la hoja',
        'Apple_scab': 'Sarna',
        'Northern_Leaf_Blight': 'Tizón foliar del norte',
        'Spider_mites Two-spotted_spider_mite': 'Ácaros araña de dos manchas',
        'Septoria_leaf_spot': 'Mancha foliar por septoria',
        'Cercospora_leaf_spot Gray_leaf_spot': 'Mancha foliar por cercospora / mancha foliar gris',
        'Leaf_scorch': 'Quemadura de la hoja'
    }

    df = build_dataset(base_dir)

    # Add Spanish species names
    df['Especie'] = df['Species'].map(species_es_map)

    # Add Spanish disease names, use 'Sano' for healthy samples
    df['Enfermedad'] = df.apply(
        lambda row: 'Sano' if row['Healthy'] else disease_es_map.get(row['Disease'], row['Disease']),
        axis=1
    )

    return df[df['Format'] == 'color']


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

