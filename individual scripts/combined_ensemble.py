import torch
import torch.nn as nn
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
import fasttext
import tempfile
import json
import hashlib

from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from io import BytesIO
from PIL import Image
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt

from tqdm import tqdm

# Constants
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_DIM = 100  # dimension for embeddings

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tok = TweetTokenizer()

class SimpleEmbedder:
    """Simple embedder that creates deterministic word vectors based on hashing.
    Used as a fallback when FastText training fails."""
    
    def __init__(self, dim=EMB_DIM):
        """Initialize with embedding dimension."""
        self.dim = dim
        self.word_vectors = {}
        
    def _get_word_vector(self, word):
        """Get vector for a single word."""
        if not word in self.word_vectors:
            # Create a deterministic vector based on word hash
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            np.random.seed(hash_val)
            self.word_vectors[word] = np.random.normal(0, 0.1, self.dim).astype(np.float32)
        return self.word_vectors[word]
        
    def get_sentence_vector(self, text):
        """Get vector for a sentence by averaging word vectors."""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return np.zeros(self.dim, dtype=np.float32)
            
        words = text.lower().split()
        if not words:
            return np.zeros(self.dim, dtype=np.float32)
            
        vectors = [self._get_word_vector(word) for word in words]
        return np.mean(vectors, axis=0)

    def get_words(self):
        """Return list of words in vocabulary."""
        return list(self.word_vectors.keys())

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
    combined_pat = r'|'.join((pat1, pat2, pat3, emotion))
    www_pat = r'www.[^ ]+'

    soup = BeautifulSoup(text, 'html.parser')
    souped = soup.get_text()
    try:
        bom_removed = souped.encode('ascii', 'ignore').decode('utf-8-sig').replace(u"\ufffd", "?")
    except:
        bom_removed = souped

    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
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

# Build FastText Models for text and caption embeddings
def build_fasttext_model(df, emb_dim=EMB_DIM, txt_variable="text", caption_variable="Image Text"):
    """
    Build a FastText model from DataFrame text data.
    
    Args:
        df: DataFrame containing text data
        emb_dim: Embedding dimension for FastText model
        txt_variable: Column name for main text
        caption_variable: Column name for caption text
    
    Returns:
        Trained FastText model or SimpleEmbedder as fallback
    """
 #   try:
    # Create a temporary file for FastText training
    ft_path = os.path.join(tempfile.gettempdir(), f"{txt_variable}_fasttext_training.txt")
    
    with open(ft_path, "w", encoding="utf-8") as ft:
        # Process text data from DataFrame for training
        if txt_variable == "text":
            # Get text data and ensure it's not None or NaN
            texts = df[txt_variable].fillna("").astype(str).tolist()
        else:  # caption
            # Get caption data and ensure it's not None or NaN
            caption_col = caption_variable if caption_variable in df.columns else "Image Text"
            texts = df[caption_col].fillna("").astype(str).tolist()
        
        # Clean the texts and write to file
        wrote_lines = 0
        for text in texts:
            cleaned_text = tweet_cleaner(text)
            if cleaned_text and len(cleaned_text.strip()) > 0:
                ft.write(cleaned_text.strip() + "\n")
                wrote_lines += 1
    
    # Check if the file has content
    file_size = os.path.getsize(ft_path)
    print(f"Created FastText training file with {file_size} bytes, {wrote_lines} lines")
    
    if file_size < 100 or wrote_lines < 10:  # Not enough data
        raise ValueError(f"Insufficient text data for {txt_variable} FastText training")
    
    # Train the FastText model
    print(f"Training FastText model for {txt_variable}...")
    model = fasttext.train_unsupervised(
    ft_path,
        model="cbow",
        dim=emb_dim,
        minCount=1,  # Include words that appear at least once
        epoch=5      # Increase epochs for better embeddings
    )
        
    # Check if model learned any embeddings
    vocab_size = len(model.get_words())
    print(f"FastText model for {txt_variable} trained with {vocab_size} words in vocabulary")
    
    # Clean up temp file
    os.remove(ft_path)
    
    if vocab_size == 0:
        raise ValueError(f"FastText model for {txt_variable} failed to learn any words")
    
    return model
        
#    except Exception as e:
#        print(f"FastText training failed for {txt_variable}: {e}")
#        print(f"Using SimpleEmbedder for {txt_variable} instead")
#        
#        # Fall back to SimpleEmbedder
#        return SimpleEmbedder(dim=emb_dim)
#                if cleaned_text and len(cleaned_text.strip()) > 0:
#                    ft.write(cleaned_text.strip() + "\n")
#                    wrote_lines += 1
#        
#        # Check if the file has content
#        file_size = os.path.getsize(ft_path)
#        print(f"Created FastText training file with {file_size} bytes, {wrote_lines} lines")
#        
#        if file_size < 100 or wrote_lines < 10:  # Not enough data
#            raise ValueError(f"Insufficient text data for {txt_variable} FastText training")
#        
#        # Train the FastText model
#        print(f"Training FastText model for {txt_variable}...")
#        model = fasttext.train_unsupervised(
#            ft_path,
#            model="cbow",
#            dim=emb_dim,
#            minCount=1,  # Include words that appear at least once
#            epoch=5      # Increase epochs for better embeddings
#        )
#        
#        # Check if model learned any embeddings
#        vocab_size = len(model.get_words())
#        print(f"FastText model for {txt_variable} trained with {vocab_size} words in vocabulary")
#        
#        # Clean up temp file
#        os.remove(ft_path)
        
#        if vocab_size == 0:
#            raise ValueError(f"FastText model for {txt_variable} failed to learn any words")
        
#        return model
        
#    except Exception as e:
#        print(f"FastText training failed for {txt_variable}: {e}")
#        print(f"Using SimpleEmbedder for {txt_variable} instead")
        
#        # Fall back to SimpleEmbedder
#        return SimpleEmbedder(dim=emb_dim)
#            if cleaned_text and len(cleaned_text.strip()) > 0:
#                ft.write(cleaned_text.strip() + "\n")
    
    # Check if the file has content
    file_size = os.path.getsize(ft_path)
    if file_size == 0:
        raise ValueError(f"No text data available for {txt_variable} in the DataFrame")
    
    print(f"Training FastText model on {file_size} bytes of {txt_variable} data...")
    
    # Train the FastText model
    model = fasttext.train_unsupervised(
        ft_path,
        model="cbow",
        dim=emb_dim,
        minCount=1,  # Include words that appear at least once
        epoch=5      # Increase epochs for better embeddings
    )
    
    # Clean up temp file
    os.remove(ft_path)
    
    # Check if model learned any embeddings
    vocab_size = len(model.get_words())
    print(f"FastText model for {txt_variable} trained with {vocab_size} words in vocabulary")
    if vocab_size == 0:
        raise ValueError(f"FastText model for {txt_variable} failed to learn any words")
    
    return model

class MyDataset(Dataset):
    def __init__(self, annotations_file, tokenizer, text_model=None, caption_model=None, max_length=128, transform=None):
        if isinstance(annotations_file, str):
            self.img_labels = pd.read_csv(annotations_file)
        elif isinstance(annotations_file, pd.DataFrame):
            self.img_labels = annotations_file
        else:
            raise ValueError("Input should be a file path or a DataFrame")
        
        self.img_labels = self.img_labels.dropna(subset=['hate', 'anti_hate'])
        self.img_labels = self.img_labels.reset_index(drop=True)
        
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_model = text_model
        self.caption_model = caption_model

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_url = self.img_labels.iloc[idx]['image']
        img_url = img_url.replace("\\", "/")
        
        # Fetch the image from the URL
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

        if valid_image and isinstance(image, Image.Image):
            image = self.transform(image)

        # Process text and image text
        text = self.img_labels.iloc[idx]['text']
        text = tweet_cleaner(text)
        
        # For the caption, use 'image_text' or 'Image Text' column based on what's available
        if 'image_text' in self.img_labels.columns:
            image_text = self.img_labels.iloc[idx]['image_text']
        else:
            image_text = self.img_labels.iloc[idx]['Image Text']
            
        image_text = tweet_cleaner(image_text)
        combined_text = str(text) + " " + str(image_text)

        # Create token inputs for transformer models
        inputs = self.tokenizer(combined_text, return_tensors="pt", max_length=self.max_length, 
                              padding='max_length', truncation=True)

        # Extract sentiment labels
        negative_sentiment = torch.tensor([self.img_labels.iloc[idx]['hate']], dtype=torch.float32)
        positive_sentiment = torch.tensor([self.img_labels.iloc[idx]['anti_hate']], dtype=torch.float32)

        # Create embeddings for text and captions if models are provided
        text_embedding = None
        caption_embedding = None
        
        if self.text_model:
            text_embedding = torch.tensor(self.text_model.get_sentence_vector(text), dtype=torch.float32)
        
        if self.caption_model:
            caption_embedding = torch.tensor(self.caption_model.get_sentence_vector(image_text), dtype=torch.float32)

        return img_url, image, combined_text, inputs, negative_sentiment, positive_sentiment, text_embedding, caption_embedding

# VisualBERT Model
class VisualBertForSentimentClassification(nn.Module):
    def __init__(self, visual_bert_model_name, dropout_rate=0.1):
        super().__init__()
        self.visual_bert = VisualBertModel.from_pretrained(visual_bert_model_name)
        self.classifier = nn.Linear(self.visual_bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids):
        outputs = self.visual_bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   visual_embeds=visual_embeds,
                                   visual_attention_mask=visual_attention_mask,
                                   visual_token_type_ids=visual_token_type_ids)

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ResNet Feature Extractor
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_features=1024):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(2048, output_features)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x

# Image preprocessing for CLIP
def preprocess_images_for_clip(images):
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    processed_images = [preprocess(image) if not torch.is_tensor(image) else image for image in images]
    return torch.stack(processed_images)

# CLIP Model for Sentiment Analysis
class CLIPForSentimentAnalysis(nn.Module):
    def __init__(self, clip_model_name, dropout_rate=0.1):
        super(CLIPForSentimentAnalysis, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, pixel_values):
        outputs = self.clip_model(input_ids=input_ids, pixel_values=pixel_values)
        logits = outputs.logits_per_image.diagonal().unsqueeze(1)
        logits = self.dropout(logits)
        logits = self.classifier(logits)
        return logits

# Fusion Model Components
class FusionModel(nn.Module):
    def __init__(self, txt_dim, img_dim, caption_dim, fusion_output_size, dropout_p=0.1):
        super(FusionModel, self).__init__()
        
        # Text and image embedders
        self.txt_module = nn.Linear(in_features=EMB_DIM, out_features=txt_dim)
        self.img_module = models.wide_resnet101_2(pretrained=True)
        self.img_module.fc = nn.Linear(in_features=2048, out_features=img_dim)
        self.caption_module = nn.Linear(in_features=EMB_DIM, out_features=caption_dim)
        
        # Fusion layer
        self.fusion = nn.Linear(
            in_features=(txt_dim + img_dim + caption_dim),
            out_features=fusion_output_size
        )
        
        # Separate classifiers for negative and positive sentiment
        self.fc_neg = nn.Linear(in_features=fusion_output_size, out_features=1)
        self.fc_pos = nn.Linear(in_features=fusion_output_size, out_features=1)
        
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, text_embedding, caption_embedding, image):
        text_features = torch.nn.functional.rrelu(self.txt_module(text_embedding))
        image_features = torch.nn.functional.rrelu(self.img_module(image))
        caption_features = torch.nn.functional.rrelu(self.caption_module(caption_embedding))
        
        combined = torch.cat([text_features, image_features, caption_features], dim=1)
        fused = self.dropout(torch.nn.functional.rrelu(self.fusion(combined)))
        
        # Output separate predictions for negative and positive sentiment
        logits_neg = self.fc_neg(fused)
        logits_pos = self.fc_pos(fused)
        
        return logits_neg, logits_pos

# Extended Ensemble Model
class ExtendedEnsembleModel(nn.Module):
    def __init__(self, visual_model_neg, visual_model_pos, clip_model_neg, clip_model_pos, 
                 fusion_model, feature_extractor, weights=None):
        super().__init__()
        self.visual_model_neg = visual_model_neg
        self.visual_model_pos = visual_model_pos
        self.clip_model_neg = clip_model_neg
        self.clip_model_pos = clip_model_pos
        self.fusion_model = fusion_model
        self.feature_extractor = feature_extractor
        
        # Initialize learnable weights for each model's contribution (6 models total)
        if weights is None:
            self.weights = nn.Parameter(torch.ones(6) / 6)  # Equal weights initially
        else:
            self.weights = nn.Parameter(torch.tensor(weights))
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, input_ids, attention_mask, clip_input_ids, pixel_values, 
                text_embeddings, caption_embeddings):
        # Process images through feature extractor for VisualBERT
        visual_embeds = self.feature_extractor(images)
        if len(visual_embeds.shape) == 2:
            visual_embeds = visual_embeds.unsqueeze(1)
            
        # Create attention masks for VisualBERT
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=images.device)
        visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long, device=images.device)
        
        # Get predictions from all models
        # VisualBERT models
        logits_visual_neg = self.visual_model_neg(input_ids, attention_mask, visual_embeds, 
                                                visual_attention_mask, visual_token_type_ids)
        logits_visual_pos = self.visual_model_pos(input_ids, attention_mask, visual_embeds, 
                                                visual_attention_mask, visual_token_type_ids)
        
        # CLIP models
        logits_clip_neg = self.clip_model_neg(input_ids=clip_input_ids, pixel_values=pixel_values)
        logits_clip_pos = self.clip_model_pos(input_ids=clip_input_ids, pixel_values=pixel_values)
        
        # Fusion model
        logits_fusion_neg, logits_fusion_pos = self.fusion_model(text_embeddings, caption_embeddings, images)
        
        # Normalize weights using softmax
        normalized_weights = torch.softmax(self.weights, dim=0)
        
        # Combine predictions using learned weights
        ensemble_neg = (normalized_weights[0] * logits_visual_neg + 
                       normalized_weights[1] * logits_clip_neg +
                       normalized_weights[2] * logits_fusion_neg)
        
        ensemble_pos = (normalized_weights[3] * logits_visual_pos + 
                       normalized_weights[4] * logits_clip_pos +
                       normalized_weights[5] * logits_fusion_pos)
        
        return ensemble_neg, ensemble_pos

# Extended Model Manager
class ExtendedEnsembleModelManager:
    def __init__(self, ensemble_model, tokenizer, clip_tokenizer, device, smoothing=0.1):
        self.ensemble_model = ensemble_model
        self.tokenizer = tokenizer
        self.clip_tokenizer = clip_tokenizer
        self.device = device
        self.criterion = BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.ensemble_model.parameters(), lr=LEARNING_RATE)
        self.scaler = GradScaler()
        self.smoothing = smoothing  # Label smoothing factor
        
    def smooth_labels(self, labels, smoothing=0.1):
        """Applies label smoothing"""
        smoothed_labels = labels * (1 - smoothing) + (smoothing / 2)
        return smoothed_labels
        
    def train_epoch(self, train_loader):
        self.ensemble_model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, total=len(train_loader), desc="Training"):
            img_path, images, texts, inputs, negative_sentiments, positive_sentiments, text_embeddings, caption_embeddings = batch
            
            # Move everything to device
            images = images.to(self.device)
            input_ids = inputs['input_ids'].squeeze(1).to(self.device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(self.device)
            negative_sentiments = negative_sentiments.float().to(self.device)
            positive_sentiments = positive_sentiments.float().to(self.device)
            text_embeddings = text_embeddings.to(self.device)
            caption_embeddings = caption_embeddings.to(self.device)
            
            # Apply label smoothing
            negative_sentiments = self.smooth_labels(negative_sentiments, self.smoothing)
            positive_sentiments = self.smooth_labels(positive_sentiments, self.smoothing)
            
            # Prepare CLIP inputs
            clip_inputs = self.clip_tokenizer(texts, padding=True, truncation=True, 
                                            return_tensors="pt").to(self.device)
            pixel_values = preprocess_images_for_clip(images).to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Use mixed precision training
            with autocast():
                # Forward pass
                ensemble_neg, ensemble_pos = self.ensemble_model(
                    images, input_ids, attention_mask, clip_inputs.input_ids, pixel_values,
                    text_embeddings, caption_embeddings
                )
                
                # Calculate loss
                loss_neg = self.criterion(ensemble_neg, negative_sentiments)
                loss_pos = self.criterion(ensemble_pos, positive_sentiments)
                loss = loss_neg + loss_pos
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, eval_loader):
        self.ensemble_model.eval()
        all_neg_preds = []
        all_pos_preds = []
        all_neg_labels = []
        all_pos_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, total=len(eval_loader), desc="Evaluating"):
                img_path, images, texts, inputs, negative_sentiments, positive_sentiments, text_embeddings, caption_embeddings = batch
                
                # Move everything to device
                images = images.to(self.device)
                input_ids = inputs['input_ids'].squeeze(1).to(self.device)
                attention_mask = inputs['attention_mask'].squeeze(1).to(self.device)
                text_embeddings = text_embeddings.to(self.device)
                caption_embeddings = caption_embeddings.to(self.device)
                
                # Prepare CLIP inputs
                clip_inputs = self.clip_tokenizer(texts, padding=True, truncation=True, 
                                                return_tensors="pt").to(self.device)
                pixel_values = preprocess_images_for_clip(images).to(self.device)
                
                # Get predictions
                ensemble_neg, ensemble_pos = self.ensemble_model(
                    images, input_ids, attention_mask, clip_inputs.input_ids, pixel_values,
                    text_embeddings, caption_embeddings
                )
                
                # Convert to probabilities
                pred_neg = torch.sigmoid(ensemble_neg).cpu().numpy()
                pred_pos = torch.sigmoid(ensemble_pos).cpu().numpy()
                
                all_neg_preds.extend(pred_neg)
                all_pos_preds.extend(pred_pos)
                all_neg_labels.extend(negative_sentiments.numpy())
                all_pos_labels.extend(positive_sentiments.numpy())
        
        # Calculate metrics
        metrics = {
            'negative': calculate_metrics(all_neg_labels, all_neg_preds),
            'positive': calculate_metrics(all_pos_labels, all_pos_preds)
        }
        
        return metrics

    def train_and_validate(self, train_loader, val_loader, num_epochs, model_save_path='best_ensemble_model.pth'):
        best_avg_f1 = 0.0
        train_losses = []
        val_metrics = {'negative_f1': [], 'positive_f1': [], 'negative_auc': [], 'positive_auc': []}
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validate
            metrics = self.evaluate(val_loader)
            
            # Track validation metrics
            val_metrics['negative_f1'].append(metrics['negative']['f1_score'])
            val_metrics['positive_f1'].append(metrics['positive']['f1_score'])
            val_metrics['negative_auc'].append(metrics['negative']['auc'])
            val_metrics['positive_auc'].append(metrics['positive']['auc'])
            
            # Calculate average F1 score
            avg_f1 = (metrics['negative']['f1_score'] + metrics['positive']['f1_score']) / 2
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Metrics:")
            print(f"Negative - F1: {metrics['negative']['f1_score']:.4f}, AUC: {metrics['negative']['auc']:.4f}")
            print(f"Positive - F1: {metrics['positive']['f1_score']:.4f}, AUC: {metrics['positive']['auc']:.4f}")
            
            # Save best model
            if avg_f1 > best_avg_f1:
                best_avg_f1 = avg_f1
                torch.save(self.ensemble_model.state_dict(), model_save_path)
                print(f"New best model saved with avg F1: {best_avg_f1:.4f}")
        
        # Plot training progress
        self.plot_training_progress(train_losses, val_metrics, num_epochs)
        
        # Load the best model for final evaluation
        self.ensemble_model.load_state_dict(torch.load(model_save_path))
        return best_avg_f1
    
    def plot_training_progress(self, train_losses, val_metrics, num_epochs):
        epochs = range(1, num_epochs + 1)
        
        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # Plot training loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot F1 scores
        plt.subplot(2, 2, 2)
        plt.plot(epochs, val_metrics['negative_f1'], 'r-', label='Negative F1')
        plt.plot(epochs, val_metrics['positive_f1'], 'g-', label='Positive F1')
        plt.title('Validation F1 Scores')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        # Plot AUC values
        plt.subplot(2, 2, 3)
        plt.plot(epochs, val_metrics['negative_auc'], 'r-', label='Negative AUC')
        plt.plot(epochs, val_metrics['positive_auc'], 'g-', label='Positive AUC')
        plt.title('Validation AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

def main(dataset_path=None, 
         model_dir='./models',
         batch_size=BATCH_SIZE,
         num_epochs=NUM_EPOCHS):
    """
    Main function to train and evaluate the ensemble model.
    Uses data directly from DataFrame as in ensemble.py.
    
    Args:
        dataset_path: Path to CSV dataset or None to use the merged datasets from ensemble.py
        model_dir: Directory to save models
        batch_size: Batch size for training
        num_epochs: Number of training epochs
    """
    # Set paths
    os.makedirs(model_dir, exist_ok=True)
    
    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load dataset from ensemble.py format
    print("Loading dataset...")
    if dataset_path:
        df = pd.read_csv(dataset_path)
    else:
        # Use the merged datasets from ensemble.py
        try:
            # Try to load from paths in the original ensemble.py
            paths = [
                '/fs/cml-scratch/kseelman/Meme_Project/multi-modal/hate/merged_final_hate_train_df.csv',
                '/fs/cml-scratch/kseelman/Meme_Project/multi-modal/hate/merged_final_hate_validation_df.csv',
                '/fs/cml-scratch/kseelman/Meme_Project/multi-modal/hate/merged_final_hate_test_df.csv'
            ]
            
            # Check if files exist before attempting to read
            if all(os.path.exists(path) for path in paths):
                df1 = pd.read_csv(paths[0])
                df1 = df1.rename(columns={
                    'img_path': 'image',
                    'image_text': 'Image Text',
                })
                df2 = pd.read_csv(paths[1])
                df3 = pd.read_csv(paths[2])
                df2 = df2.rename(columns={
                    'img_path': 'image',
                    'image_text': 'Image Text',
                })
                df3 = df3.rename(columns={
                    'img_path': 'image',
                    'image_text': 'Image Text',
                })
                df = pd.concat([df1, df2, df3], ignore_index=True)
            else:
                raise FileNotFoundError("One or more ensemble.py dataset files not found")
                
        except Exception as e:
            print(f"Error loading ensemble.py datasets: {e}")
            print("Falling back to X5k dataset...")
            # Fall back to X5k dataset - adjust this path as needed
            x5k_path = '/fs/cml-scratch/kseelman/Meme_Project/multi-modal/data/X5k/X5k_all_with_caption.csv'
            if os.path.exists(x5k_path):
                df = pd.read_csv(x5k_path)
            else:
                # Last resort - create a small synthetic dataset for testing
                print("X5k dataset not found, creating synthetic data for testing")
                synthetic_data = {
                    'text': ['Text ' + str(i) for i in range(100)],
                    'image': ['http://example.com/img' + str(i) + '.jpg' for i in range(100)],
                    'Image Text': ['Caption ' + str(i) for i in range(100)],
                    'hate': np.random.binomial(1, 0.3, 100),
                    'anti_hate': np.random.binomial(1, 0.3, 100)
                }
                df = pd.DataFrame(synthetic_data)
                # Ensure hate and anti_hate are mutually exclusive
                for i in range(len(df)):
                    if df.loc[i, 'hate'] == 1 and df.loc[i, 'anti_hate'] == 1:
                        df.loc[i, 'anti_hate'] = 0
    
    # Print dataset info
    print(f"Dataset loaded with {len(df)} entries")
    print(f"Column names: {df.columns.tolist()}")
    
    if len(df) > 0:
        print(f"Sample text: {df['text'].iloc[0]}")
        print(f"Sample image path: {df['image'].iloc[0]}")
    
    # Ensure required columns exist
    required_columns = ['text', 'image', 'hate', 'anti_hate']
    caption_col = 'Image Text' if 'Image Text' in df.columns else 'image_text'
    if caption_col not in df.columns:
        print(f"Warning: No caption column found. Creating empty '{caption_col}' column.")
        df[caption_col] = ""
    
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Dataset missing required columns: {missing}. Found columns: {df.columns.tolist()}")
    
    # Ensure hate and anti_hate are properly formatted
    df['hate'] = df['hate'].astype(float)
    df['anti_hate'] = df['anti_hate'].astype(float)
    
    # Build FastText models
    print("Building embedders for text and captions...")
    try:
        # Identify caption column name
        print(f"Using '{caption_col}' as caption column")
        
        # Try to build FastText models
        text_model = build_fasttext_model(df, EMB_DIM, "text", caption_col)
        caption_model = build_fasttext_model(df, EMB_DIM, caption_col, caption_col)
    except Exception as e:
        print(f"Failed to build embedders: {e}")
        print("Using SimpleEmbedder fallback...")
        
        # Create SimpleEmbedder instances as fallback
        text_model = SimpleEmbedder(dim=EMB_DIM)
        caption_model = SimpleEmbedder(dim=EMB_DIM)
        
        # Pre-train the embedders on the dataset to create vocabulary
        print("Pre-training SimpleEmbedder with dataset vocabulary...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building vocabulary"):
            text = tweet_cleaner(str(row['text'])) if 'text' in df.columns else ""
            caption = tweet_cleaner(str(row[caption_col])) if caption_col in df.columns else ""
            
            if text:
                _ = text_model.get_sentence_vector(text)
            if caption:
                _ = caption_model.get_sentence_vector(caption)
        
        print(f"SimpleEmbedder vocabulary sizes - Text: {len(text_model.get_words())}, Caption: {len(caption_model.get_words())}")
    # Create dataset
    dataset = MyDataset(
        df,
        tokenizer,
        text_model=text_model,
        caption_model=caption_model,
        transform=transform
    )
    
    # Calculate class weights for balanced training
    subset_negative_labels = df['hate'].values
    subset_positive_labels = df['anti_hate'].values
    
    num_negative = np.sum(subset_negative_labels)
    num_positive = np.sum(subset_positive_labels)
    
    if num_negative == 0 or num_positive == 0:
        print(f"Warning: One of the classes has zero samples. num_negative: {num_negative}, num_positive: {num_positive}")
        num_negative = max(1, num_negative)
        num_positive = max(1, num_positive)
    
    weight_negative = num_positive / (num_negative + num_positive)
    weight_positive = num_negative / (num_negative + num_positive)
    
    print(f"Class weights - Negative: {weight_negative}, Positive: {weight_positive}")
    
    # Split dataset into train, validation, and test sets
    total_size = len(dataset)
    train_size = int(0.60 * total_size)
    val_size = int(0.20 * total_size)
    test_size = total_size - train_size - val_size
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    print(f"Dataset split: Train={train_size}, Validation={val_size}, Test={test_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize models
    print("Initializing models...")
    # VisualBERT models
    visual_model_neg = VisualBertForSentimentClassification("uclanlp/visualbert-nlvr2-coco-pre").to(DEVICE)
    visual_model_pos = VisualBertForSentimentClassification("uclanlp/visualbert-nlvr2-coco-pre").to(DEVICE)
    
    # CLIP models
    clip_model_neg = CLIPForSentimentAnalysis("openai/clip-vit-base-patch32").to(DEVICE)
    clip_model_pos = CLIPForSentimentAnalysis("openai/clip-vit-base-patch32").to(DEVICE)
    
    # Feature extractor
    feature_extractor = ResNetFeatureExtractor().to(DEVICE)
    
    # Fusion model
    fusion_model = FusionModel(
        txt_dim=200,
        img_dim=224,
        caption_dim=200,
        fusion_output_size=200,
        dropout_p=0.1
    ).to(DEVICE)
    
    # Load pre-trained models if available
    try:
        model_files = {
            'visual_neg': f'{model_dir}/best_visual_model_hate.pth',
            'visual_pos': f'{model_dir}/best_visual_model_anti_hate.pth',
            'clip_neg': f'{model_dir}/best_clip_model_hate.pth',
            'clip_pos': f'{model_dir}/best_clip_model_anti_hate.pth'
        }
        
        # Check if model files exist
        existing_models = [k for k, v in model_files.items() if os.path.exists(v)]
        if existing_models:
            print(f"Found pre-trained models: {existing_models}")
            
            if 'visual_neg' in existing_models:
                visual_model_neg.load_state_dict(torch.load(model_files['visual_neg']))
            if 'visual_pos' in existing_models:
                visual_model_pos.load_state_dict(torch.load(model_files['visual_pos']))
            if 'clip_neg' in existing_models:
                clip_model_neg.load_state_dict(torch.load(model_files['clip_neg']))
            if 'clip_pos' in existing_models:
                clip_model_pos.load_state_dict(torch.load(model_files['clip_pos']))
                
            print("Pre-trained models loaded successfully.")
        else:
            print("No pre-trained models found. Starting from scratch.")
    except Exception as e:
        print(f"Warning: Could not load pre-trained models. Starting from scratch. Error: {e}")
    
    # Create ensemble model
    ensemble_model = ExtendedEnsembleModel(
        visual_model_neg=visual_model_neg,
        visual_model_pos=visual_model_pos,
        clip_model_neg=clip_model_neg,
        clip_model_pos=clip_model_pos,
        fusion_model=fusion_model,
        feature_extractor=feature_extractor
    ).to(DEVICE)
    
    # Create ensemble manager
    ensemble_manager = ExtendedEnsembleModelManager(
        ensemble_model=ensemble_model,
        tokenizer=tokenizer,
        clip_tokenizer=clip_tokenizer,
        device=DEVICE
    )
    
    # Update criterion with class weights
    ensemble_manager.criterion = criterion
    
    # Train the ensemble model
    print("Training ensemble model...")
    best_avg_f1 = ensemble_manager.train_and_validate(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        model_save_path=f'{model_dir}/best_ensemble_model.pth'
    )
    print(f"Training completed with best average F1 score: {best_avg_f1:.4f}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    ensemble_model.load_state_dict(torch.load(f'{model_dir}/best_ensemble_model.pth'))
    test_metrics = ensemble_manager.evaluate(test_loader)
    
    print("Test Set Results:")
    print(f"Negative Sentiment:")
    print(f"  - F1 Score: {test_metrics['negative']['f1_score']:.4f}")
    print(f"  - AUC: {test_metrics['negative']['auc']:.4f}")
    print(f"  - Accuracy: {test_metrics['negative']['accuracy']:.4f}")
    print(f"  - Precision: {test_metrics['negative']['precision']:.4f}")
    print(f"  - Recall: {test_metrics['negative']['recall']:.4f}")
    
    print(f"Positive Sentiment:")
    print(f"  - F1 Score: {test_metrics['positive']['f1_score']:.4f}")
    print(f"  - AUC: {test_metrics['positive']['auc']:.4f}")
    print(f"  - Accuracy: {test_metrics['positive']['accuracy']:.4f}")
    print(f"  - Precision: {test_metrics['positive']['precision']:.4f}")
    print(f"  - Recall: {test_metrics['positive']['recall']:.4f}")
    
    # Save final results
    results = {
        'test_metrics': test_metrics,
        'best_val_f1': best_avg_f1,
        'model_path': f'{model_dir}/best_ensemble_model.pth'
    }
    
    with open(f'{model_dir}/ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    return ensemble_model, ensemble_manager, test_metrics

if __name__ == "__main__":
    # Example usage with custom arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate ensemble model with fusion')
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset CSV')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=6, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Update constants if provided
    if args.lr != LEARNING_RATE:
        LEARNING_RATE = args.lr
        print(f"Using custom learning rate: {LEARNING_RATE}")
    
    # Run main function with parsed arguments
    main(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs
    )
