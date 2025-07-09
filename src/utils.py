import os
import kaggle

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
