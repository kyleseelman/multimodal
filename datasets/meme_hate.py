# Meme hate dataset for modular use

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO

class MemeHateDataset(Dataset):
    def __init__(self, annotations_file, topic_dist, tokenizer, max_length=128, transform=None, augment_transform=None):
        if isinstance(annotations_file, str):
            self.img_labels = pd.read_csv(annotations_file)
        elif isinstance(annotations_file, pd.DataFrame):
            self.img_labels = annotations_file
        else:
            raise ValueError("Input should be a file path or a DataFrame")
        self.img_labels = self.img_labels.dropna(subset=['hate', 'anti_hate'])
        self.img_labels = self.img_labels.reset_index(drop=True)
        self.topic_dist = topic_dist
        self.transform = transform
        self.augment_transform = augment_transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_url = self.img_labels.loc[idx, 'image']
        img_url = img_url.replace("\\", "/")
        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            if "image" not in response.headers["Content-Type"]:
                raise ValueError(f"URL {img_url} does not contain an image")
            image = Image.open(BytesIO(response.content)).convert("RGB")
            valid_image = True
        except (requests.exceptions.RequestException, ValueError, UnidentifiedImageError) as e:
            print(f"Error fetching or processing image from {img_url}: {e}")
            image = torch.zeros(3, 224, 224)
            valid_image = False
        if valid_image and isinstance(image, Image.Image):
            image = self.transform(image) if self.transform else image
        text = self.img_labels.loc[idx, 'text']
        image_text = self.img_labels.loc[idx, 'Image Text'] if 'Image Text' in self.img_labels.columns else ''
        text = str(text) + " " + str(image_text)
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding='max_length', truncation=True)
        negative_sentiment = torch.tensor([self.img_labels.loc[idx, 'hate']], dtype=torch.float32)
        positive_sentiment = torch.tensor([self.img_labels.loc[idx, 'anti_hate']], dtype=torch.float32)
        return img_url, image, text, inputs, negative_sentiment, positive_sentiment, self.topic_dist[idx] 