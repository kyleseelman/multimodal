import torch
import torch.nn as nn
# from torch.nn import MSELoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import AdaptiveAvgPool2d
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import AdaptiveAvgPool2d, Linear, CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler

from torchvision.models import resnet50
from torchvision import transforms, models
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import VisualBertModel, AutoTokenizer, CLIPModel, CLIPTokenizer, AdamW

import torchvision.transforms as transforms

import os
import requests
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import TweetTokenizer
import math

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


from io import BytesIO
from PIL import Image
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt


from tqdm import tqdm
torch.cuda.empty_cache()

# Constants
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly changes the brightness, contrast, saturation, and hue of the image by up to 20%.
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tok = TweetTokenizer()

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese characters
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\u3030"
        "\ufe0f"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def tweet_cleaner(text):

    # Check if text is None or NaN, or not a string
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    if not isinstance(text, str):
        text = str(text)

    pat1 = r'@[A-Za-z0-9_]+'
    pat2 = r'https?://[^ ]+'
    pat3 = r'\\'
    emotion = r'[:;]+["^-]*[()]+'
    combined_pat = r'|'.join((pat1, pat2, pat3, emotion)) #, emotion
    www_pat = r'www.[^ ]+'

    soup = BeautifulSoup(text, 'html.parser')
    souped = soup.get_text()
    try:
        bom_removed = souped.encode('ascii', 'ignore').decode('utf-8-sig').replace(u"\ufffd", "?")
    except:
        bom_removed = souped

    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    #stripped = remove_emojis(stripped)
    lower_case = stripped.lower()
    letters_only = re.sub("[^a-zA-Z]", " ", lower_case)
    words = [x for x in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()

def calculate_metrics(actual, predicted_probs, threshold=0.5):
    # Flatten the actual labels from the array format
    actual = np.array([a[0] for a in actual])

    # Convert predicted probabilities to binary predictions using the threshold
    predicted_labels = (np.array(predicted_probs) >= threshold).astype(int)

    # Calculate different metrics
    accuracy = accuracy_score(actual, predicted_labels)
    precision = precision_score(actual, predicted_labels, zero_division=0)
    recall = recall_score(actual, predicted_labels, zero_division=0)
    f1 = f1_score(actual, predicted_labels, zero_division=0)

    # AUC calculation (requires raw probabilities, not binary labels)
    auc = roc_auc_score(actual, predicted_probs)

    # Return a dictionary with all metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

class MyDataset_Copy(Dataset):
    def __init__(self, annotations_file, topic_dist, tokenizer, max_length=128, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels.dropna(subset=['hate', 'anti_hate'])
        self.img_labels = self.img_labels.reset_index(drop=True)

        self.topic_dist = topic_dist

        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_url = self.img_labels.loc[idx, 'image_url']
        img_url = img_url.replace("\\", "/")

        # Fetch the image from the URL (storing the images in github)
        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()

            # Check if the content is an image
            if "image" not in response.headers["Content-Type"]:
                raise ValueError(f"URL {img_url} does not contain an image")

            # Try to open the image using PIL
            image = Image.open(BytesIO(response.content)).convert("RGB")
            valid_image = True
        except (requests.exceptions.RequestException, ValueError, UnidentifiedImageError) as e:
            print(f"Error fetching or processing image from {img_url}: {e}")
            # Use a zero-filled tensor as a placeholder image
            image = torch.zeros(3, 224, 224)
            valid_image = False

        # Apply transformation only if the image is valid and it's a PIL Image
        if valid_image and isinstance(image, Image.Image):
            image = self.transform(image)

        text = self.img_labels.loc[idx, 'text']
        text = tweet_cleaner(text)
        image_text = self.img_labels.loc[idx, 'image_text']
        image_text = tweet_cleaner(image_text)
        text = str(text) + " " + str(image_text)

        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding='max_length', truncation=True)

        negative_sentiment = torch.tensor([self.img_labels.loc[idx, 'hate']], dtype=torch.float32)
        positive_sentiment = torch.tensor([self.img_labels.loc[idx, 'anti_hate']], dtype=torch.float32)

        return img_url, image, text, inputs, negative_sentiment, positive_sentiment, topic_dist[idx]

class MyDataset(Dataset):
    def __init__(self, annotations_file, topic_dist, tokenizer, max_length=128, transform=None, augment_transform=None):
        if isinstance(annotations_file, str):
            # Load data from CSV if a file path is provided
            self.img_labels = pd.read_csv(annotations_file)
        elif isinstance(annotations_file, pd.DataFrame):
            # Directly assign data if a DataFrame is provided
            self.img_labels = annotations_file
        else:
            raise ValueError("Input should be a file path or a DataFrame")




        self.img_labels = self.img_labels.dropna(subset=['hate', 'anti_hate'])
        self.img_labels = self.img_labels.reset_index(drop=True)

        self.topic_dist = topic_dist

        self.transform = transform  # General transform for all images
        self.augment_transform = augment_transform  # Specific augment transform for "hate" class
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_url = self.img_labels.loc[idx, 'image']
        img_url = img_url.replace("\\", "/")

        # Fetch the image from the URL (storing the images in GitHub)
        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()

            # Check if the content is an image
            if "image" not in response.headers["Content-Type"]:
                raise ValueError(f"URL {img_url} does not contain an image")

            # Try to open the image using PIL
            image = Image.open(BytesIO(response.content)).convert("RGB")
            valid_image = True
        except (requests.exceptions.RequestException, ValueError, UnidentifiedImageError) as e:
            print(f"Error fetching or processing image from {img_url}: {e}")
            # Use a zero-filled tensor as a placeholder image
            image = torch.zeros(3, 224, 224)
            valid_image = False

        # Apply the general transformation if the image is a valid PIL image
        if valid_image and isinstance(image, Image.Image):
            image = self.transform(image)

        # Apply augmentations only if it's still a PIL image and needs augmentation
        #if self.img_labels.loc[idx, 'hate'] == 1 and valid_image and isinstance(image, Image.Image):
         #   image = self.augment_transform(image)

        #elif self.img_labels.loc[idx, 'anti_hate'] == 1 and valid_image and isinstance(image, Image.Image):
        #    image = self.augment_transform(image, class_label_column='anti_hate')

        # If image is already a tensor (like the zero-filled tensor), skip transformations
#        elif isinstance(image, torch.Tensor):
#            pass  # Skip augmentations if image is already a tensor (e.g., placeholder image)

        # Tokenizing the text and image captions
        text = self.img_labels.loc[idx, 'text']
        text = tweet_cleaner(text)
        image_text = self.img_labels.loc[idx, 'Image Text']
        image_text = tweet_cleaner(image_text)
        text = str(text) + " " + str(image_text)

        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding='max_length', truncation=True)

        # Labels for hate and anti-hate
        negative_sentiment = torch.tensor([self.img_labels.loc[idx, 'hate']], dtype=torch.float32)
        positive_sentiment = torch.tensor([self.img_labels.loc[idx, 'anti_hate']], dtype=torch.float32)

        return img_url, image, text, inputs, negative_sentiment, positive_sentiment, self.topic_dist[idx]

    # Add filter method to filter dataset by a list of indices
    def filter_by_indices(self, indices):
        filtered_img_labels = self.img_labels.iloc[indices].reset_index(drop=True)
        # Return a new instance of MyDataset with filtered img_labels
        return MyDataset(filtered_img_labels, self.topic_dist, self.tokenizer, self.max_length, self.transform, self.augment_transform)


# Define your standard and augment transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Apply color jittering
    transforms.RandomHorizontalFlip(),  # Random flipping
    transforms.RandomRotation(degrees=15),  # Random rotation
    transforms.ToTensor(),
])

class VisualBertForSentimentClassification(nn.Module):
    """"This class initializes a VisualBERT model for sentiment classification.
    It includes a VisualBERT model, a linear classifier, a dropout layer for regularization,
    and a sigmoid activation function.
    The forward method processes input through the VisualBERT model and
    then through the classifier. It returns logits which can be transformed into
    probabilities using a sigmoid function (commented out in your code)."""
    def __init__(self, visual_bert_model_name, dropout_rate=0.1):

        super().__init__()
        self.visual_bert = VisualBertModel.from_pretrained(visual_bert_model_name)
        # Classifier layer
        self.classifier = nn.Linear(self.visual_bert.config.hidden_size+20, 1)  # Outputting a single score
        #  Dropout layer to prevent overfitting: randomly set a fraction `dropout_rate` of input units to 0 during training
        self.dropout = nn.Dropout(dropout_rate)
        # Adding a sigmoid activation function- turns row score into probs bn 0 and 1
        self.sigmoid = nn.Sigmoid()

# take inputs and process through visual_bert and then classifier and sigmoid function to get the final probability
    def forward(self, input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids, topic_dist):
        outputs = self.visual_bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   visual_embeds=visual_embeds,
                                   visual_attention_mask=visual_attention_mask,
                                   visual_token_type_ids=visual_token_type_ids)

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)  # Apply dropout for overfitting
        
        combined_features = torch.cat([pooled_output, topic_dist.float().to(DEVICE)], dim=1)

        # Get raw logits
        logits = self.classifier(combined_features)

        return logits
        # return self.sigmoid(logits) # suitable for regression-like tasks with continuous labels

# Feature Extractor
class ResNetFeatureExtractor(nn.Module):
    """This class serves as a feature extractor using a ResNet model, specifically ResNet50.
        It extracts features from images and projects them to a specified output size to match
        the input size required by VisualBERT. The forward method processes an image through the ResNet model,
        applies average pooling, flattens the output, and then projects it to the desired output features size."""

    def __init__(self, output_features=1024):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2]) #selects the layers from this model
        self.pool = AdaptiveAvgPool2d((1, 1)) #resizes the output feature maps produced by the convolutional layers of the neural network.
        self.proj = nn.Linear(2048, output_features)  # Project to match VisualBERT input

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.proj(x)  # Apply projection (a linear projection layer is used to map the image patch “arrays” to patch embedding “vectors”.)
        return x

feature_extractor = ResNetFeatureExtractor().to(DEVICE)

"""This function preprocesses a list of images to be compatible with the CLIP model.
The preprocessing steps include resizing, center cropping, converting to a tensor,
and normalizing using CLIP-specific mean and standard deviation values.
The function handles both PIL images and pre-converted PyTorch tensors.
Processed images are stacked together and returned as a single tensor."""

# prepare image for clip model
def preprocess_images_for_clip(images):
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    processed_images = [preprocess(image) if not torch.is_tensor(image) else image for image in images]
    return torch.stack(processed_images)


"""This class integrates the CLIP model for sentiment analysis.
It includes a dropout layer for regularization and a classifier layer consisting of a linear transformation followed by a sigmoid activation function.
The forward method passes input text and pixel values through the CLIP model, extracts the diagonal elements of the output logits (representing image-text compatibility), applies dropout and the classifier, and finally applies sigmoid activation to obtain probabilities.
The probabilities are transformed into sentiment labels (0 or 1) based on a threshold of 0.5."""
# processes the text and images using CLIP, gets a score for each image-text pair, and then converts these scores into probabilities
class CLIPForSentimentAnalysis(nn.Module):
    def __init__(self, clip_model_name, dropout_rate=0.1):
        super(CLIPForSentimentAnalysis, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.dropout = nn.Dropout(dropout_rate)

        self.classifier = nn.Linear(1+20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, pixel_values, topic_dist):
        # Pass inputs through CLIP model
        outputs = self.clip_model(input_ids=input_ids, pixel_values=pixel_values)

        # Extract the diagonal elements representing the image-text compatibility
        logits = outputs.logits_per_image.diagonal().unsqueeze(1)
        logits = self.dropout(logits)  # Apply dropout before classifier
        
        combined_features = torch.cat([logits, topic_dist.float().to(DEVICE)], dim=1)

        # Apply classifier
        logits = self.classifier(combined_features)
        # Apply sigmoid activation to get probabilities
        probs = self.sigmoid(logits)
        # Convert probabilities to sentiment labels (0 or 1)
        labels = (probs > 0.5).int()

        return logits

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        """
        :param gamma: Focusing parameter for Focal Loss. Default is 2.
        :param alpha: Weighting factor for the class imbalance. If specified, should be a tensor.
        :param reduction: Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert logits to probabilities using sigmoid
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt is the probability that the model correctly classifies

        # Focal Loss adjustment: (1 - pt)^gamma
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        # Apply alpha weighting for class imbalance
        if self.alpha is not None:
            alpha = self.alpha.gather(0, targets.long().view(-1))
            F_loss = alpha * F_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, augmentation_transforms, class_label='hate', num_augmentations=2):
        """
        Parameters:
        - original_dataset: Original MyDataset instance
        - augmentation_transforms: List of transform compositions to apply
        - class_label: Column name for the minority class ('hate' or 'anti_hate')
        - num_augmentations: Number of augmented versions to create for each minority sample
        """
        self.original_dataset = original_dataset
        self.augmentation_transforms = augmentation_transforms
        self.class_label = class_label
        self.num_augmentations = num_augmentations
        self.transform = original_dataset.transform
        self.tokenizer = original_dataset.tokenizer
        self.max_length = original_dataset.max_length
        
        # Create augmented DataFrame
        self._create_augmented_data()
        
    def _create_augmented_data(self):
        # Get the original DataFrame
        original_df = self.original_dataset.img_labels.copy()
        
        # Find minority class samples
        minority_samples = original_df[original_df[self.class_label] == 1]
        
        # Create list to store augmented rows
        augmented_rows = []
        
        print(f"Creating {self.num_augmentations} augmented versions for {len(minority_samples)} minority samples...")
        for idx, row in tqdm(minority_samples.iterrows()):
            img_url = row['image'].replace("\\", "/")
            
            # Create augmented versions
            for aug_idx in range(self.num_augmentations):
                # Create a copy of the row for augmentation
                aug_row = row.copy()
                aug_row['is_augmented'] = True
                aug_row['original_idx'] = idx
                aug_row['aug_version'] = aug_idx
                augmented_rows.append(aug_row)
        
        # Create DataFrame with augmented samples
        augmented_df = pd.DataFrame(augmented_rows)
        
        # Combine original and augmented DataFrames
        self.img_labels = pd.concat([original_df, augmented_df], ignore_index=True)
        
        # Store which indices are augmented
        self.augmented_indices = set(range(len(original_df), len(self.img_labels)))
        
        # If topic distributions exist, handle them
        if hasattr(self.original_dataset, 'topic_dist'):
            original_topic_dist = self.original_dataset.topic_dist
            augmented_topic_dist = np.vstack([original_topic_dist[row['original_idx']] 
                                            for idx, row in augmented_df.iterrows()])
            self.topic_dist = np.vstack([original_topic_dist, augmented_topic_dist])
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]
        img_url = row['image'].replace("\\", "/")
        
        # Check if this is an augmented sample
        is_augmented = idx in self.augmented_indices
        
        try:
            # Fetch image
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Apply augmentation if it's an augmented sample
            if is_augmented:
                transform = np.random.choice(self.augmentation_transforms)
                image = transform(image)
            
            # Apply standard transformation
            if self.transform:
                image = self.transform(image)
                
        except Exception as e:
            print(f"Error processing image from {img_url}: {e}")
            image = torch.zeros(3, 224, 224)  # Default size matching your transform
        
        # Prepare text input
        text = row['text']
        image_text = row['Image Text']
        text = str(text) + " " + str(image_text)
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        # Get labels
        negative_sentiment = torch.tensor([row['hate']], dtype=torch.float32)
        positive_sentiment = torch.tensor([row['anti_hate']], dtype=torch.float32)
        
        # Get topic distribution if it exists
        topic_dist = self.topic_dist[idx] if hasattr(self, 'topic_dist') else None
        
        return img_url, image, text, inputs, negative_sentiment, positive_sentiment, topic_dist

def create_augmentation_transforms():
    """Create a list of different augmentation combinations"""
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    
    transform_list = [
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            color_jitter
        ]),
        transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            color_jitter
        ]),
        transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            color_jitter
        ])
    ]
    return transform_list

# Usage example
def create_augmented_dataset(original_dataset, num_augmentations=2):
    """
    Create an augmented dataset with balanced classes
    
    Parameters:
    - original_dataset: Original MyDataset instance
    - num_augmentations: Number of augmented versions to create for each minority sample
    
    Returns:
    - Augmented dataset instance
    """
    augmentation_transforms = create_augmentation_transforms()
    augmented_dataset = AugmentedDataset(
        original_dataset,
        augmentation_transforms,
        class_label='hate',  # or 'anti_hate' depending on which is minority
        num_augmentations=num_augmentations
    )
    
    return augmented_dataset

#print(weight_negative, weight_positive)

# create class that handles validation and training of models
class ModelManager:
    def __init__(self, visual_model_neg, visual_model_pos, clip_model_neg, clip_model_pos, feature_extractor, tokenizer, clip_tokenizer, loss_fn, optimizer_visual_neg, optimizer_visual_pos, optimizer_clip_neg, optimizer_clip_pos, smoothing=0.1):
        # Initialization with models, tokenizers, loss function, and optimizers
        self.visual_model_neg = visual_model_neg
        self.visual_model_pos = visual_model_pos
        self.clip_model_neg = clip_model_neg
        self.clip_model_pos = clip_model_pos
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.clip_tokenizer = clip_tokenizer

        # Calculate class imbalance weights
        self.positive_weight = None  # Will be calculated based on the dataset

        # Initialize BCEWithLogitsLoss with pos_weight later after dataset analysis
        self.loss_fn = loss_fn

#        self.loss_fn = FocalLoss(gamma=2, alpha=torch.tensor([0.2, 0.8]).to(DEVICE), reduction='mean')  # Use class_weights if provided

        self.optimizer_visual_neg = optimizer_visual_neg
        self.optimizer_visual_pos = optimizer_visual_pos
        self.optimizer_clip_neg = optimizer_clip_neg
        self.optimizer_clip_pos = optimizer_clip_pos
        self.scaler = GradScaler()  # Move scaler initialization here
        self.visual_neg_losses = []
        self.visual_pos_losses = []
        self.clip_neg_losses = []
        self.clip_pos_losses = []
        self.smoothing = smoothing  # Smoothing factor for label smoothing

    def smooth_labels(self, labels, smoothing=0.1):
        """Applies label smoothing"""
        smoothed_labels = labels * (1 - smoothing) + (smoothing / 2)
        return smoothed_labels

    def validate(self, dataloader, sentiment_type):
        # Select the correct models based on sentiment type
        if sentiment_type == 'hate':
            visual_model = self.visual_model_neg
            clip_model = self.clip_model_neg
        else:
            visual_model = self.visual_model_pos
            clip_model = self.clip_model_pos

        # Set models to evaluation mode
        visual_model.eval()
        clip_model.eval()
        self.feature_extractor.eval()

        total_loss_visual, total_loss_clip = 0, 0

        with torch.no_grad():
            for batch, (image_path, images, texts, inputs, negative_sentiments, positive_sentiments, topic_dist) in tqdm(
                    enumerate(dataloader), total=len(dataloader), desc=f"Validating {sentiment_type}"):

                # Prepare inputs
                images = images.to(DEVICE)
                input_ids = inputs['input_ids'].squeeze(1).to(DEVICE)
                attention_mask = inputs['attention_mask'].squeeze(1).to(DEVICE)
                labels = (negative_sentiments.float() if sentiment_type == 'hate' else positive_sentiments.float()).to(DEVICE)

                # Apply label smoothing
                labels = self.smooth_labels(labels, smoothing=self.smoothing)

                # Process images through feature extractor
                visual_embeds = self.feature_extractor(images)
                if len(visual_embeds.shape) == 2:
                    visual_embeds = visual_embeds.unsqueeze(1)

                # VisualBERT forward pass
                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=DEVICE)
                visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long, device=DEVICE)
                logits_visual = visual_model(input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids, topic_dist)

                # Calculate loss for VisualBERT
                loss_visual = self.loss_fn(logits_visual, labels)
                total_loss_visual += loss_visual.item()

                # CLIP forward pass
                text_inputs = self.clip_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                outputs_clip = clip_model(input_ids=text_inputs.input_ids, pixel_values=preprocess_images_for_clip(images), topic_dist=topic_dist)

                # Calculate loss for CLIP
                loss_clip = self.loss_fn(outputs_clip, labels)
                total_loss_clip += loss_clip.item()

        avg_loss_visual = total_loss_visual / len(dataloader)
        avg_loss_clip = total_loss_clip / len(dataloader)

        return avg_loss_visual, avg_loss_clip

    def train(self, dataloader, sentiment_type, epoch_number):
        # Choose the appropriate models and optimizers based on sentiment type
        visual_model = self.visual_model_neg if sentiment_type == 'hate' else self.visual_model_pos
        clip_model = self.clip_model_neg if sentiment_type == 'hate' else self.clip_model_pos
        optimizer_visual = self.optimizer_visual_neg if sentiment_type == 'hate' else self.optimizer_visual_pos
        optimizer_clip = self.optimizer_clip_neg if sentiment_type == 'hate' else self.optimizer_clip_pos

        # Set models to training mode
        visual_model.train()
        clip_model.train()

        # Training loop
        for batch, (image_path, images, texts, inputs, negative_sentiments, positive_sentiments, topic_dist) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch_number+1} [Training {sentiment_type}]"):
            images = images.to(DEVICE)
            input_ids = inputs['input_ids'].squeeze(1).to(DEVICE)
            attention_mask = inputs['attention_mask'].squeeze(1).to(DEVICE)
            labels = (negative_sentiments.float() if sentiment_type == 'hate' else positive_sentiments.float()).to(DEVICE)

            # Apply label smoothing
            labels = self.smooth_labels(labels, smoothing=self.smoothing)

            # Zero gradients for both optimizers
            optimizer_visual.zero_grad()
            optimizer_clip.zero_grad()

            # Process images through feature extractor
            with torch.no_grad():
                visual_embeds = self.feature_extractor(images)
            if len(visual_embeds.shape) == 2:
                visual_embeds = visual_embeds.unsqueeze(1)

            # Mixed precision training with autocast
            with autocast():
                # VisualBERT model forward pass
                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=DEVICE)
                visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long, device=DEVICE)
                logits_visual = visual_model(input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids, topic_dist)

                # Calculate loss for VisualBERT
                loss_visual = self.loss_fn(logits_visual, labels)

                # CLIP model forward pass
                text_inputs = self.clip_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                outputs_clip = clip_model(input_ids=text_inputs.input_ids, pixel_values=preprocess_images_for_clip(images), topic_dist=topic_dist)
                loss_clip = self.loss_fn(outputs_clip, labels)

            # Scale loss and backward
            self.scaler.scale(loss_visual).backward()
            self.scaler.scale(loss_clip).backward()

            # Update parameters using scaled gradients
            self.scaler.step(optimizer_visual)
            self.scaler.step(optimizer_clip)

            # Update the scaler
            self.scaler.update()

    def train_and_validate(self, train_loader, val_loader, num_epochs):
        """integrates both training and validation for a specified number of epochs, saving the best models based on validation loss."""

        best_loss_visual_neg = float('inf')
        best_loss_clip_neg = float('inf')
        best_loss_visual_pos = float('inf')
        best_loss_clip_pos = float('inf')

        for epoch in range(num_epochs):
            # Train for one epoch for each sentiment type
            self.train(train_loader, 'hate', epoch)
            self.train(train_loader, 'anti_hate', epoch)

            # Validate for each sentiment type
            val_loss_visual_neg, val_loss_clip_neg = self.validate(val_loader, 'hate')
            val_loss_visual_pos, val_loss_clip_pos = self.validate(val_loader, 'anti_hate')

            # Track validation losses
            self.visual_neg_losses.append(val_loss_visual_neg)
            self.visual_pos_losses.append(val_loss_visual_pos)
            self.clip_neg_losses.append(val_loss_clip_neg)
            self.clip_pos_losses.append(val_loss_clip_pos)

            # Save best models for negative sentiment
            if val_loss_visual_neg < best_loss_visual_neg:
                best_loss_visual_neg = val_loss_visual_neg
                torch.save(self.visual_model_neg.state_dict(), '/fs/cml-scratch/kseelman/Meme_Project/multi-modal/models/best_visual_model_hate10_topic_sin_focal.pth')

            if val_loss_clip_neg < best_loss_clip_neg:
                best_loss_clip_neg = val_loss_clip_neg
                torch.save(self.clip_model_neg.state_dict(), '/fs/cml-scratch/kseelman/Meme_Project/multi-modal/models/best_clip_model_hate10_topic_sin_focal.pth')

            # Save best models for positive sentiment
            if val_loss_visual_pos < best_loss_visual_pos:
                best_loss_visual_pos = val_loss_visual_pos
                torch.save(self.visual_model_pos.state_dict(), '/fs/cml-scratch/kseelman/Meme_Project/multi-modal/models/best_visual_model_anti_hate10_topic_sin_focal.pth')  #update path as needed

            if val_loss_clip_pos < best_loss_clip_pos:
                best_loss_clip_pos = val_loss_clip_pos
                torch.save(self.clip_model_pos.state_dict(), '/fs/cml-scratch/kseelman/Meme_Project/multi-modal/models/best_clip_model_anti_hate10_topic_sin_focal.pth')  #update path as needed

            # Print validation losses for tracking
            print(f" Validation Losses - VisualBERT Negative: {val_loss_visual_neg}, CLIP Negative: {val_loss_clip_neg}, VisualBERT Positive: {val_loss_visual_pos}, CLIP Positive: {val_loss_clip_pos}")

    def plot_validation_loss(self):
        """Plot the validation losses over the epochs."""
        epochs = range(1, len(self.visual_neg_losses) + 1)
        plt.plot(epochs, self.visual_neg_losses, 'r', label='VisualBERT Negative')
        plt.plot(epochs, self.visual_pos_losses, 'g', label='VisualBERT Positive')
        plt.plot(epochs, self.clip_neg_losses, 'b', label='CLIP Negative')
        plt.plot(epochs, self.clip_pos_losses, 'y', label='CLIP Positive')
        plt.title('Validation Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def evaluate_test_set(model_manager, test_loader):
        # Load the best models
        model_manager.visual_model_neg.load_state_dict(torch.load('/fs/cml-scratch/kseelman/Meme_Project/multi-modal/models/best_visual_model_hate10_topic_sin_focal.pth'))  #update path as needed
        model_manager.visual_model_pos.load_state_dict(torch.load('/fs/cml-scratch/kseelman/Meme_Project/multi-modal/models/best_visual_model_anti_hate10_topic_sin_focal.pth')) #update path as needed
        model_manager.clip_model_neg.load_state_dict(torch.load('/fs/cml-scratch/kseelman/Meme_Project/multi-modal/models/best_clip_model_hate10_topic_sin_focal.pth')) #update path as needed
        model_manager.clip_model_pos.load_state_dict(torch.load('/fs/cml-scratch/kseelman/Meme_Project/multi-modal/models/best_clip_model_anti_hate10_topic_sin_focal.pth'))  #update path as needed

        sigmoid = torch.nn.Sigmoid()  # Define sigmoid function for converting logits to probabilities
        data = {
            'image_name': [], 'text': [], 'actual_positive_label': [], 'actual_negative_label': [],
            'predicted_probs_visualbert_positive': [], 'predicted_probs_visualbert_negative': [],
            'predicted_probs_clip_positive': [], 'predicted_probs_clip_negative': []
        }

        with torch.no_grad():
            for image_path, images, texts, inputs, negative_sentiments, positive_sentiments, topic_dist in test_loader:
                images = images.to(DEVICE)
                input_ids = inputs['input_ids'].squeeze(1).to(DEVICE)
                attention_mask = inputs['attention_mask'].squeeze(1).to(DEVICE)

                # Process images through feature extractor
                visual_embeds = model_manager.feature_extractor(images)
                if len(visual_embeds.shape) == 2:
                    visual_embeds = visual_embeds.unsqueeze(1)
                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=DEVICE)
                visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long, device=DEVICE)

                # Get logits from VisualBERT models and convert to probabilities
                logits_visual_neg = model_manager.visual_model_neg(input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids, topic_dist)
                logits_visual_pos = model_manager.visual_model_pos(input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids, topic_dist)
                probs_visual_neg = sigmoid(logits_visual_neg).squeeze().cpu()
                probs_visual_pos = sigmoid(logits_visual_pos).squeeze().cpu()

                # Prepare texts for CLIP model
                text_inputs = model_manager.clip_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

                # Get logits from CLIP models and convert to probabilities
                logits_clip_neg = model_manager.clip_model_neg(input_ids=text_inputs.input_ids, pixel_values=preprocess_images_for_clip(images), topic_dist=topic_dist)
                logits_clip_pos = model_manager.clip_model_pos(input_ids=text_inputs.input_ids, pixel_values=preprocess_images_for_clip(images), topic_dist=topic_dist)
                probs_clip_neg = sigmoid(logits_clip_neg).squeeze().cpu()
                probs_clip_pos = sigmoid(logits_clip_pos).squeeze().cpu()

                # Append predictions and labels to data
                data['image_name'].extend(image_path)
                data['text'].extend(texts)
                data['actual_positive_label'].extend(positive_sentiments.numpy())
                data['actual_negative_label'].extend(negative_sentiments.numpy())
                data['predicted_probs_visualbert_positive'].extend(probs_visual_pos.numpy())
                data['predicted_probs_visualbert_negative'].extend(probs_visual_neg.numpy())
                data['predicted_probs_clip_positive'].extend(probs_clip_pos.numpy())
                data['predicted_probs_clip_negative'].extend(probs_clip_neg.numpy())
        #print(data['predicted_probs_visualbert_positive'])
        #print(data['predicted_probs_clip_positive'])
        # Calculate and return evaluation metrics
        metrics = {
            'VisualBERT_Positive': calculate_metrics(data['actual_positive_label'], data['predicted_probs_visualbert_positive']),
            'VisualBERT_Negative': calculate_metrics(data['actual_negative_label'], data['predicted_probs_visualbert_negative']),
            'CLIP_Positive': calculate_metrics(data['actual_positive_label'], data['predicted_probs_clip_positive']),
            'CLIP_Negative': calculate_metrics(data['actual_negative_label'], data['predicted_probs_clip_negative'])
        }
        print(metrics)
        return pd.DataFrame(data), metrics, probs_visual_neg, probs_visual_pos

def augment_dataset(dataset, augment_transform, class_label_column='hate'):
    # Filter the dataset to get only minority class samples (hate class)
    minority_class = dataset[dataset[class_label_column] == 1]

    # Create an empty list to store augmented data
    augmented_data = []

    for idx, row in minority_class.iterrows():
        img_url = row['image'].replace("\\", "/")

        # Fetch the image from the URL
        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()

            # Check if the content is an image
            if "image" in response.headers["Content-Type"]:
                image = Image.open(BytesIO(response.content)).convert("RGB")
                # Apply augmentations
                augmented_image = augment_transform(image)

                # Save the augmented image locally or to a list (depending on your pipeline)
                augmented_data.append({
                    'image': img_url,  # You can modify this if saving new image files
                    'text': row['text'],
                    'Image Text': row['Image Text'],
                    'hate': row['hate'],
                    'anti_hate': row['anti_hate'],
                })

        except (requests.exceptions.RequestException, ValueError, UnidentifiedImageError) as e:
            print(f"Error fetching or processing image from {img_url}: {e}")

    # Convert augmented data to DataFrame and concatenate it with original data
    augmented_df = pd.DataFrame(augmented_data)
    dataset = pd.concat([dataset, augmented_df], ignore_index=True)
    print(f"Dataset size after augmentation: {len(dataset)}")

    return dataset

def topic_modeling(data):
    text_dataset = (data['text'] + " " + data['Image Text']).tolist()
    text_dataset = [tweet_cleaner(text) for text in text_dataset]

    # Download NLTK stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Step 3: Tokenize and Remove Stopwords
    def remove_stopwords(text):
        tokens = text.split()  # Tokenize by splitting text into words
        tokens = [word for word in tokens if word.lower() not in stop_words]  # Remove stopwords
        return " ".join(tokens)  # Join the tokens back into a single string

    # Apply stopword removal to all documents
    documents_cleaned = [remove_stopwords(doc) for doc in text_dataset]

    # Step 4: Fit the topic model (LDA, for example) on the cleaned text data
    vectorizer = CountVectorizer(max_features=1000)  # You can adjust the max_features based on your data
    X_train_texts = vectorizer.fit_transform(documents_cleaned)  # Vectorize the cleaned text

    lda_model = LatentDirichletAllocation(n_components=20, random_state=42)
    train_topic_distributions = lda_model.fit_transform(X_train_texts)

    return train_topic_distributions


print("--------- Loading and Creating Dataset ----------")
# Combine CSVs
df1 = pd.read_csv('/fs/cml-scratch/kseelman/Meme_Project/multi-modal/data/X5k/X5k_all_with_caption.csv')
df2 = pd.read_csv('/fs/cml-scratch/kseelman/Meme_Project/multi-modal/data/sample_annotations.csv')
df2 = df2.rename(columns={
    'image_url': 'image',
    'image_text': 'Image Text'
})
# Concatenate the DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)
#combined_df = augment_dataset(combined_df, augment_transform)
topic_dist = topic_modeling(combined_df)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load dataset
#dataset = MyDataset('/content/drive/MyDrive/Memes Project Aim 1/training_data/X5k/X5k_all_with_caption.csv', tokenizer, transform=transform)

#dataset = MyDataset(
#    '/fs/cml-scratch/kseelman/Meme_Project/multi-modal/data/X5k/X5k_all_with_caption.csv',
#    tokenizer,
#    transform=transform,
#    augment_transform=augment_transform
#)
dataset = MyDataset(
    combined_df,
    topic_dist,
    tokenizer,
    transform=transform,
    augment_transform=augment_transform
)
print(type(dataset[0]))

# # Apply the augmentation to the dataset and increase size
#dataset = augment_dataset(dataset, augment_transform)
#print(type(dataset[0]))
#balanced_dataset = create_balanced_dataset(dataset, target_size=500)


# Print dataset size after dropping NaN
print(f"Total dataset size after dropping NaN: {len(dataset)}")

# Create augmented dataset
augmented_dataset = create_augmented_dataset(dataset, num_augmentations=2)

print(f"Total dataset size after augmentation: {len(augmented_dataset)}")

# Split the dataset
total_size = len(augmented_dataset)
train_size = int(0.60 * total_size)
val_size = int(0.20 * total_size)
test_size = total_size - train_size - val_size

print(train_size, test_size)

# Split the dataset into training, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(augmented_dataset, [train_size, val_size, test_size])
# Calculate positive class weight based on the training split dataset
#train_img_labels = dataset.img_labels.loc[train_dataset.indices]  # Access the img_labels of the training set
#positive_weight = len(train_img_labels['hate']) / len(train_img_labels['anti_hate'])
#print(len(train_img_labels['hate']), len(train_img_labels['anti_hate']))
# Pass this as a parameter to BCEWithLogitsLoss to account for class imbalance
loss_fn = BCEWithLogitsLoss().to(DEVICE)

# Create balanced sampler function
def create_balanced_sampler(subset, original_dataset):
    subset_indices = subset.indices
    subset_negative_labels = original_dataset.img_labels.loc[subset_indices, 'hate'].values
    subset_positive_labels = original_dataset.img_labels.loc[subset_indices, 'anti_hate'].values

    num_negative = np.sum(subset_negative_labels)
    num_positive = np.sum(subset_positive_labels)

    if num_negative == 0 or num_positive == 0:
        print(f"Warning: One of the classes has zero samples. num_negative: {num_negative}, num_positive: {num_positive}")
        num_negative = max(1, num_negative)
        num_positive = max(1, num_positive)

    weight_negative = num_positive / (num_negative + num_positive)
    weight_positive = num_negative / (num_negative + num_positive)

    sample_weights = np.where(subset_negative_labels == 1, weight_negative, weight_positive)
    return WeightedRandomSampler(sample_weights, len(sample_weights)), weight_negative, weight_positive

# Create samplers and DataLoaders
train_sampler, weight_negative, weight_positive = create_balanced_sampler(train_dataset, augmented_dataset)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)



df = pd.read_csv('/fs/cml-scratch/kseelman/Meme_Project/multi-modal/data/X5k/X5k_all_with_caption.csv')

print(df.hate.value_counts(), df.anti_hate.value_counts())

print(df.positive.value_counts(), df.negative.value_counts())


# Initialize Models for Negative and Positive Sentiments
visual_model_neg = VisualBertForSentimentClassification("uclanlp/visualbert-nlvr2-coco-pre").to(DEVICE)
visual_model_pos = VisualBertForSentimentClassification("uclanlp/visualbert-nlvr2-coco-pre").to(DEVICE)

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model_neg = CLIPForSentimentAnalysis("openai/clip-vit-base-patch32").to(DEVICE)
clip_model_pos = CLIPForSentimentAnalysis("openai/clip-vit-base-patch32").to(DEVICE)

# Initialize Optimizers for Each Model
optimizer_visual_neg = AdamW(visual_model_neg.parameters(), lr=LEARNING_RATE)
optimizer_visual_pos = AdamW(visual_model_pos.parameters(), lr=LEARNING_RATE)
optimizer_clip_neg = AdamW(clip_model_neg.parameters(), lr=LEARNING_RATE)
optimizer_clip_pos = AdamW(clip_model_pos.parameters(), lr=LEARNING_RATE)

# Create Model Manager Instance
model_manager = ModelManager(
    visual_model_neg, visual_model_pos,
    clip_model_neg, clip_model_pos,
    feature_extractor, tokenizer, clip_tokenizer,
    loss_fn, optimizer_visual_neg, optimizer_visual_pos, optimizer_clip_neg, optimizer_clip_pos
)


def train(model_manager, train_loader, val_loader):
    # Training Loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}\n-------------------------------")
        model_manager.train_and_validate(train_loader, val_loader, 1)

    print("Training done!")

def test(model_manager, test_loader, new=False):

    if new:
    # Comment out for new test set
        df = pd.read_csv('/fs/cml-scratch/kseelman/Meme_Project/multi-modal/data/latest_annotations.csv')
        df_new = df.rename(columns={
            'image_url': 'image',
            'image_text': 'Image Text'
        })
        topic_dist = topic_modeling(df_new)

        new_test_set = MyDataset_Copy('/fs/cml-scratch/kseelman/Meme_Project/multi-modal/data/latest_annotations.csv', topic_dist, tokenizer, transform=transform)
        new_test_loader = DataLoader(new_test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model_manager.evaluate_test_set(new_test_loader)

    else:
        model_manager.evaluate_test_set(test_loader)
#train(model_manager, train_loader, val_loader)
test(model_manager, test_loader, True)


