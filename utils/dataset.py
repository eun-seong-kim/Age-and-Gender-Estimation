from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms
import warnings


class ImageDataset(Dataset):
    def __init__(self, df, params, transform=None, aug_transform=None, aug_multiplier=3):
        self.df = df
        self.params = params
        self.image_paths = df['path'].values
        self.gender = df['gender'].values
        self.age = df['age'].values
        self.transform = transform
        self.aug_transform = aug_transform
        self.aug_multiplier = aug_multiplier

        # Define underrepresented age groups for augmentation
        self.aug_indices = [
            i for i in range(len(df))
            if (self.age[i] < 10) or (20 < self.age[i] < 45) or (self.age[i] > 50)
        ]

    def __len__(self):
        if self.aug_transform:
            return len(self.df) + len(self.aug_indices) * self.aug_multiplier
        return len(self.df)

    def __getitem__(self, idx):
        if idx < len(self.df):
            return self._load_image_and_label(idx, self.transform)
        else:
            aug_idx = idx - len(self.df)
            orig_idx = self.aug_indices[aug_idx // self.aug_multiplier]
            return self._load_image_and_label(orig_idx, self.aug_transform)

    def _load_image_and_label(self, idx, transform):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError):
            warnings.warn(f"[Warning] Error opening image: {img_path}")
            img = torch.zeros(3, self.params['img_size'], self.params['img_size'])
            labels = torch.tensor([0.0, 0.0], dtype=torch.float32)
            return img, labels

        age = self.age[idx]
        gender = self.gender[idx]
        labels = torch.tensor([age, gender], dtype=torch.float32)

        if transform:
            img = transform(img)
        else:
            img = transforms.Resize((self.params['img_size'], self.params['img_size']))(img)
            img = transforms.ToTensor()(img)

        return img, labels
