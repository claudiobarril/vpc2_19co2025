import torch
import os
import cv2 as cv


class PlantVillageDataset(torch.utils.data.Dataset):
    def __init__(self, df, root_dir, format_type='color', transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.format_type = format_type
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder = f"{row['Species']}___{'healthy' if row['Healthy'] else row['Disease']}"
        image_path = os.path.join(self.root_dir, folder, row['FileName'])

        image = cv.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Convertir BGR a RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # image es numpy array uint8, listo para ToPILImage
        label = torch.tensor(row['Label_id'], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label