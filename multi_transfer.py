import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import AdaptiveAvgPool2d
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F

from torchvision.models import resnet50
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from transformers import (
    VisualBertModel, 
    AutoTokenizer, 
    CLIPModel, 
    CLIPTokenizer, 
    AdamW
)

import os
import requests
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import TweetTokenizer
import math
from PIL import Image
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt
from tqdm import tqdm
from io import BytesIO
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

# Constants
BATCH_SIZE = 16
LEARNING_RATE = 5e-5  # Reduced from 1e-4
NUM_EPOCHS_PHASE1 = 6
NUM_EPOCHS_PHASE2 = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Text cleaning utilities
tok = TweetTokenizer()

def tweet_cleaner(text):
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
    """Calculate various classification metrics"""
    actual = np.array(actual).reshape(-1)
    predicted_probs = np.array(predicted_probs).reshape(-1)
    predicted_labels = (predicted_probs >= threshold).astype(int)
    
    try:
        accuracy = accuracy_score(actual, predicted_labels)
        precision = precision_score(actual, predicted_labels, zero_division=0)
        recall = recall_score(actual, predicted_labels, zero_division=0)
        f1 = f1_score(actual, predicted_labels, zero_division=0)
        
        if len(np.unique(actual)) < 2:
            auc = 0.0
        else:
            auc = roc_auc_score(actual, predicted_probs)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        # Add confusion matrix counts
        true_positives = ((predicted_labels == 1) & (actual == 1)).sum()
        false_positives = ((predicted_labels == 1) & (actual == 0)).sum()
        true_negatives = ((predicted_labels == 0) & (actual == 0)).sum()
        false_negatives = ((predicted_labels == 0) & (actual == 1)).sum()
        
        metrics.update({
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        })
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        print(f"Shape of actual: {actual.shape}")
        print(f"Shape of predicted_probs: {predicted_probs.shape}")
        print(f"Unique values in actual: {np.unique(actual)}")
        print(f"Range of predicted_probs: [{predicted_probs.min()}, {predicted_probs.max()}]")
        return None

class MyDataset(Dataset):
    def __init__(self, annotations_file, tokenizer, max_length=128, transform=None):
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

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_url = self.img_labels.iloc[idx]['image']
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
            image = self.transform(image)

        text = self.img_labels.iloc[idx]['text']
        text = tweet_cleaner(text)
        image_text = self.img_labels.iloc[idx]['Image Text']
        image_text = tweet_cleaner(image_text)
        text = str(text) + " " + str(image_text)

        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, 
                              padding='max_length', truncation=True)

        negative_sentiment = torch.tensor([self.img_labels.iloc[idx]['hate']], dtype=torch.float32)
        positive_sentiment = torch.tensor([self.img_labels.iloc[idx]['anti_hate']], dtype=torch.float32)

        return img_url, image, text, inputs, negative_sentiment, positive_sentiment

class VisualBertForSentimentClassification(nn.Module):
    def __init__(self, visual_bert_model_name, dropout_rate=0.3):
        super().__init__()
        self.visual_bert = VisualBertModel.from_pretrained(visual_bert_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.visual_bert.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.l2_reg = 0.01

    def forward(self, input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids):
        outputs = self.visual_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def get_l2_reg_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.l2_reg * l2_loss

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
        return self.proj(x)

class CLIPForSentimentAnalysis(nn.Module):
    def __init__(self, clip_model_name, dropout_rate=0.3):
        super(CLIPForSentimentAnalysis, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        self.l2_reg = 0.01

    def forward(self, input_ids, pixel_values):
        outputs = self.clip_model(input_ids=input_ids, pixel_values=pixel_values)
        logits = outputs.logits_per_image.diagonal().unsqueeze(1)
        logits = self.dropout(logits)
        return self.classifier(logits)

    def get_l2_reg_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.l2_reg * l2_loss

def preprocess_images_for_clip(images):
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    processed_images = [preprocess(image) if not torch.is_tensor(image) else image for image in images]
    return torch.stack(processed_images)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class ModelManager:
    def __init__(self, visual_model_neg, visual_model_pos, clip_model_neg, clip_model_pos, 
                 feature_extractor, tokenizer, clip_tokenizer, loss_fn, 
                 optimizer_visual_neg, optimizer_visual_pos, optimizer_clip_neg, optimizer_clip_pos, 
                 smoothing=0.1):
        self.visual_model_neg = visual_model_neg
        self.visual_model_pos = visual_model_pos
        self.clip_model_neg = clip_model_neg
        self.clip_model_pos = clip_model_pos
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.clip_tokenizer = clip_tokenizer
        self.loss_fn = loss_fn
        self.optimizer_visual_neg = optimizer_visual_neg
        self.optimizer_visual_pos = optimizer_visual_pos
        self.optimizer_clip_neg = optimizer_clip_neg
        self.optimizer_clip_pos = optimizer_clip_pos
        self.scaler = GradScaler()
        self.smoothing = smoothing
       
        # Create Focal Loss instances
        self.loss_fn_hate = FocalLoss(alpha=0.25, gamma=2.0).to(DEVICE)
        self.loss_fn_antihate = FocalLoss(alpha=0.25, gamma=2.0).to(DEVICE)

        # Add schedulers
        self.scheduler_visual_neg = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_visual_neg, mode='min', factor=0.5, patience=2, verbose=True
        )
        self.scheduler_visual_pos = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_visual_pos, mode='min', factor=0.5, patience=2, verbose=True
        )
        self.scheduler_clip_neg = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_clip_neg, mode='min', factor=0.5, patience=2, verbose=True
        )
        self.scheduler_clip_pos = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_clip_pos, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        self.phase1_losses = {'visual_neg': [], 'visual_pos': [], 'clip_neg': [], 'clip_pos': []}
        self.phase2_losses = {'visual_neg': [], 'visual_pos': [], 'clip_neg': [], 'clip_pos': []}

    def smooth_labels(self, labels, smoothing=0.1):
        return labels * (1 - smoothing) + (smoothing / 2)

    def train_single_epoch(self, dataloader, sentiment_type, epoch, phase):
        visual_model = self.visual_model_neg if sentiment_type == 'hate' else self.visual_model_pos
        clip_model = self.clip_model_neg if sentiment_type == 'hate' else self.clip_model_pos
        optimizer_visual = self.optimizer_visual_neg if sentiment_type == 'hate' else self.optimizer_visual_pos
        optimizer_clip = self.optimizer_clip_neg if sentiment_type == 'hate' else self.optimizer_clip_pos
        loss_fn = self.loss_fn_hate if sentiment_type == 'hate' else self.loss_fn_antihate

        visual_model.train()
        clip_model.train()
        self.feature_extractor.train()

        total_loss_visual = 0
        total_loss_clip = 0

        for batch_idx, (image_path, images, texts, inputs, negative_sentiments, positive_sentiments) in enumerate(
                tqdm(dataloader, desc=f"Phase {phase} - Epoch {epoch+1} Training {sentiment_type}")):
            
            images = images.to(DEVICE)
            input_ids = inputs['input_ids'].squeeze(1).to(DEVICE)
            attention_mask = inputs['attention_mask'].squeeze(1).to(DEVICE)
            labels = (negative_sentiments.float() if sentiment_type == 'hate' else positive_sentiments.float()).to(DEVICE)
            
            labels = self.smooth_labels(labels, self.smoothing)

            optimizer_visual.zero_grad()
            optimizer_clip.zero_grad()

            with torch.no_grad():
                visual_embeds = self.feature_extractor(images)
            if len(visual_embeds.shape) == 2:
                visual_embeds = visual_embeds.unsqueeze(1)

            with autocast():
                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=DEVICE)
                visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long, device=DEVICE)
                
                # Forward passes
                logits_visual = visual_model(input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids)
                text_inputs = self.clip_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                outputs_clip = clip_model(input_ids=text_inputs.input_ids, pixel_values=preprocess_images_for_clip(images))
                
                # Calculate losses with regularization
                loss_visual = loss_fn(logits_visual, labels) + visual_model.get_l2_reg_loss()
                loss_clip = loss_fn(outputs_clip, labels) + clip_model.get_l2_reg_loss()

            # Gradient clipping
            self.scaler.scale(loss_visual).backward()
            self.scaler.scale(loss_clip).backward()
            
            torch.nn.utils.clip_grad_norm_(visual_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm=1.0)

            self.scaler.step(optimizer_visual)
            self.scaler.step(optimizer_clip)
            self.scaler.update()

            total_loss_visual += loss_visual.item()
            total_loss_clip += loss_clip.item()

        return total_loss_visual / len(dataloader), total_loss_clip / len(dataloader)

    def validate(self, dataloader):
        self.visual_model_neg.eval()
        self.visual_model_pos.eval()
        self.clip_model_neg.eval()
        self.clip_model_pos.eval()
        self.feature_extractor.eval()

        losses = {
            'visual_neg': 0, 'visual_pos': 0,
            'clip_neg': 0, 'clip_pos': 0
        }

        with torch.no_grad():
            for batch_idx, (image_path, images, texts, inputs, negative_sentiments, positive_sentiments) in enumerate(dataloader):
                images = images.to(DEVICE)
                input_ids = inputs['input_ids'].squeeze(1).to(DEVICE)
                attention_mask = inputs['attention_mask'].squeeze(1).to(DEVICE)
                neg_labels = negative_sentiments.float().to(DEVICE)
                pos_labels = positive_sentiments.float().to(DEVICE)

                visual_embeds = self.feature_extractor(images)
                if len(visual_embeds.shape) == 2:
                    visual_embeds = visual_embeds.unsqueeze(1)

                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=DEVICE)
                visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long, device=DEVICE)

                logits_visual_neg = self.visual_model_neg(input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids)
                logits_visual_pos = self.visual_model_pos(input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids)
                
                text_inputs = self.clip_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                outputs_clip_neg = self.clip_model_neg(input_ids=text_inputs.input_ids, pixel_values=preprocess_images_for_clip(images))
                outputs_clip_pos = self.clip_model_pos(input_ids=text_inputs.input_ids, pixel_values=preprocess_images_for_clip(images))

                losses['visual_neg'] += self.loss_fn(logits_visual_neg, neg_labels).item()
                losses['visual_pos'] += self.loss_fn(logits_visual_pos, pos_labels).item()
                losses['clip_neg'] += self.loss_fn(outputs_clip_neg, neg_labels).item()
                losses['clip_pos'] += self.loss_fn(outputs_clip_pos, pos_labels).item()

        for key in losses:
            losses[key] /= len(dataloader)

        return losses

    def train_phase1(self, train_loader, val_loader, num_epochs):
        """First training phase on balanced dataset"""
        print("Starting Phase 1: Training on balanced dataset")
        best_losses = {
            'visual_neg': float('inf'), 'visual_pos': float('inf'),
            'clip_neg': float('inf'), 'clip_pos': float('inf')
        }

        for epoch in range(num_epochs):
            visual_loss_neg, clip_loss_neg = self.train_single_epoch(train_loader, 'hate', epoch, phase=1)
            visual_loss_pos, clip_loss_pos = self.train_single_epoch(train_loader, 'anti_hate', epoch, phase=1)

            val_losses = self.validate(val_loader)
            
            # Update schedulers
            self.scheduler_visual_neg.step(val_losses['visual_neg'])
            self.scheduler_visual_pos.step(val_losses['visual_pos'])
            self.scheduler_clip_neg.step(val_losses['clip_neg'])
            self.scheduler_clip_pos.step(val_losses['clip_pos'])

            for model_type in ['visual_neg', 'visual_pos', 'clip_neg', 'clip_pos']:
                self.phase1_losses[model_type].append(val_losses[model_type])
                
                if val_losses[model_type] < best_losses[model_type]:
                    best_losses[model_type] = val_losses[model_type]
                    self.save_model(model_type, 'phase1')

            print(f"Phase 1 - Epoch {epoch+1}/{num_epochs} Validation Losses:", val_losses)

    def train_phase2(self, hate_minority_loader, antihate_minority_loader, val_loader, num_epochs, learning_rate=1e-5):
        """Second training phase focusing on minority classes"""
        print("Starting Phase 2: Fine-tuning on minority classes")
        
        self.load_best_models('phase1')
        self.adjust_learning_rate(learning_rate)
        
        best_losses = {
            'visual_neg': float('inf'), 'visual_pos': float('inf'),
            'clip_neg': float('inf'), 'clip_pos': float('inf')
        }

        for epoch in range(num_epochs):
            # Train on hate minority samples
            print(f"\nEpoch {epoch+1} - Training on hate minority samples")
            visual_loss_neg, clip_loss_neg = self.train_single_epoch(
                hate_minority_loader, 'hate', epoch, phase=2
            )
            
            # Train on anti-hate minority samples
            print(f"Epoch {epoch+1} - Training on anti-hate minority samples")
            visual_loss_pos, clip_loss_pos = self.train_single_epoch(
                antihate_minority_loader, 'anti_hate', epoch, phase=2
            )

            # Validate
            val_losses = self.validate(val_loader)
            
            # Update schedulers
            self.scheduler_visual_neg.step(val_losses['visual_neg'])
            self.scheduler_visual_pos.step(val_losses['visual_pos'])
            self.scheduler_clip_neg.step(val_losses['clip_neg'])
            self.scheduler_clip_pos.step(val_losses['clip_pos'])
            
            for model_type in ['visual_neg', 'visual_pos', 'clip_neg', 'clip_pos']:
                self.phase2_losses[model_type].append(val_losses[model_type])
                
                if val_losses[model_type] < best_losses[model_type]:
                    best_losses[model_type] = val_losses[model_type]
                    self.save_model(model_type, 'phase2')

            print(f"Phase 2 - Epoch {epoch+1}/{num_epochs} Validation Losses:", val_losses)

    def adjust_learning_rate(self, new_lr):
        """Adjust learning rate for all optimizers"""
        for param_group in self.optimizer_visual_neg.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.optimizer_visual_pos.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.optimizer_clip_neg.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.optimizer_clip_pos.param_groups:
            param_group['lr'] = new_lr

    def save_model(self, model_type, phase):
        """Save model checkpoints"""
        model_map = {
            'visual_neg': self.visual_model_neg,
            'visual_pos': self.visual_model_pos,
            'clip_neg': self.clip_model_neg,
            'clip_pos': self.clip_model_pos
        }
        
        path = f'/fs/cml-scratch/kseelman/Meme_Project/multi-modal/models/best_{model_type}_{phase}.pth'
        torch.save(model_map[model_type].state_dict(), path)

    def load_best_models(self, phase):
        """Load best models from specified phase"""
        model_map = {
            'visual_neg': self.visual_model_neg,
            'visual_pos': self.visual_model_pos,
            'clip_neg': self.clip_model_neg,
            'clip_pos': self.clip_model_pos
        }
        
        for model_type, model in model_map.items():
            path = f'/fs/cml-scratch/kseelman/Meme_Project/multi-modal/models/best_{model_type}_{phase}.pth'
            model.load_state_dict(torch.load(path))

    def plot_training_progress(self):
        """Plot training progress for both phases"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for model_type in self.phase1_losses:
            ax1.plot(self.phase1_losses[model_type], label=model_type)
        ax1.set_title('Phase 1: Balanced Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        for model_type in self.phase2_losses:
            ax2.plot(self.phase2_losses[model_type], label=model_type)
        ax2.set_title('Phase 2: Minority Class Fine-tuning')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def evaluate_test_set(self, test_loader):
        """Evaluate with threshold optimization"""
        print("\nEvaluating models on test set...")
        
        self.load_best_models('phase2')
        
        # Set models to evaluation mode
        self.visual_model_neg.eval()
        self.visual_model_pos.eval()
        self.clip_model_neg.eval()
        self.clip_model_pos.eval()
        self.feature_extractor.eval()
        
        # Collect predictions and labels
        predictions = {
            'visual_neg': [], 'visual_pos': [],
            'clip_neg': [], 'clip_pos': []
        }
        actual = {'hate': [], 'anti_hate': []}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                image_path, images, texts, inputs, negative_sentiments, positive_sentiments = batch
                
                # Process batch
                images = images.to(DEVICE)
                input_ids = inputs['input_ids'].squeeze(1).to(DEVICE)
                attention_mask = inputs['attention_mask'].squeeze(1).to(DEVICE)
                
                # Get model predictions
                visual_embeds = self.feature_extractor(images)
                if len(visual_embeds.shape) == 2:
                    visual_embeds = visual_embeds.unsqueeze(1)
                
                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=DEVICE)
                visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long, device=DEVICE)
                
                # Get predictions
                pred_visual_neg = torch.sigmoid(self.visual_model_neg(
                    input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids
                )).cpu().numpy()
            
                pred_visual_pos = torch.sigmoid(self.visual_model_pos(
                    input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids
                )).cpu().numpy()
                
                text_inputs = self.clip_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                pred_clip_neg = torch.sigmoid(self.clip_model_neg(
                    input_ids=text_inputs.input_ids, pixel_values=preprocess_images_for_clip(images)
                )).cpu().numpy()
                
                pred_clip_pos = torch.sigmoid(self.clip_model_pos(
                    input_ids=text_inputs.input_ids, pixel_values=preprocess_images_for_clip(images)
                )).cpu().numpy()
                
                # Store predictions and actual labels
                predictions['visual_neg'].extend(pred_visual_neg)
                predictions['visual_pos'].extend(pred_visual_pos)
                predictions['clip_neg'].extend(pred_clip_neg)
                predictions['clip_pos'].extend(pred_clip_pos)
                actual['hate'].extend(negative_sentiments.numpy())
                actual['anti_hate'].extend(positive_sentiments.numpy())
    
    # Find optimal thresholds
        thresholds = {model: [] for model in predictions.keys()}
        for model in predictions.keys():
            best_f1 = 0
            best_threshold = 0.5
            for threshold in np.arange(0.1, 0.9, 0.05):
                pred_labels = (np.array(predictions[model]) >= threshold).astype(int)
                actual_labels = actual['hate'] if 'neg' in model else actual['anti_hate']
                f1 = f1_score(actual_labels, pred_labels)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            thresholds[model] = best_threshold
        
        # Calculate metrics with optimal thresholds
        metrics = {}
        for model in predictions.keys():
            actual_labels = actual['hate'] if 'neg' in model else actual['anti_hate']
            metrics[model] = calculate_metrics(
                actual_labels,
                predictions[model],
            threshold=thresholds[model]
            )
    
        print("\nOptimal Thresholds:")
        for model, threshold in thresholds.items():
            print(f"{model}: {threshold:.3f}")
        
        print("\nTest Set Metrics:")
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name}:")
            for metric_name, value in model_metrics.items():
                print(f"{metric_name}: {value:.4f}")
        
        return predictions, actual, metrics, thresholds

def create_dataloaders(dataset, batch_size=8):
    """Create dataloaders handling nested class imbalance"""
    # First split the dataset
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size]
    )
    
    # Get indices for minority classes (class "1") in both hate and antihate
    train_indices = train_dataset.indices
    train_df = dataset.img_labels.iloc[train_indices]
    
    hate_minority_indices = [idx for idx in train_indices 
                           if dataset.img_labels.iloc[idx]['hate'] == 1]
    antihate_minority_indices = [idx for idx in train_indices 
                                if dataset.img_labels.iloc[idx]['anti_hate'] == 1]
    
    # Create minority datasets for fine-tuning
    hate_minority_dataset = Subset(dataset, hate_minority_indices)
    antihate_minority_dataset = Subset(dataset, antihate_minority_indices)
    
    # Calculate weights for balanced sampling in the first phase
    weights = []
    for idx in train_indices:
        sample = dataset.img_labels.iloc[idx]
        if sample['hate'] == 1 or sample['anti_hate'] == 1:
            weights.append(2.0)  # Higher weight for minority class
        else:
            weights.append(1.0)  # Lower weight for majority class
    
    balanced_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=balanced_sampler,
        num_workers=4,
        drop_last=True
    )
    
    hate_minority_loader = DataLoader(
        hate_minority_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    antihate_minority_loader = DataLoader(
        antihate_minority_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )
    
    print("\nDataset Statistics:")
    print(f"Total dataset size: {total_size}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Hate minority (class 1) size: {len(hate_minority_dataset)}")
    print(f"Anti-hate minority (class 1) size: {len(antihate_minority_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    return train_loader, hate_minority_loader, antihate_minority_loader, val_loader, test_loader

def train_two_phase(dataset, num_epochs_phase1=6, num_epochs_phase2=3):
    """Main training function with improved training strategy"""
    print("Initializing two-phase training...")
    
    # Create dataloaders
    train_loader, hate_minority_loader, antihate_minority_loader, val_loader, test_loader = create_dataloaders(
        dataset, 
        batch_size=BATCH_SIZE
    )
    
    # Calculate class weights based on class distribution
    hate_pos = len([1 for idx in range(len(dataset)) if dataset.img_labels.iloc[idx]['hate'] == 1])
    hate_neg = len(dataset) - hate_pos
    antihate_pos = len([1 for idx in range(len(dataset)) if dataset.img_labels.iloc[idx]['anti_hate'] == 1])
    antihate_neg = len(dataset) - antihate_pos
    
    hate_weight = hate_neg / hate_pos
    antihate_weight = antihate_neg / antihate_pos
    
    print(f"\nClass weights:")
    print(f"Hate positive weight: {hate_weight:.2f}")
    print(f"Anti-hate positive weight: {antihate_weight:.2f}")
    
    # Initialize models with improved architecture
    visual_model_neg = VisualBertForSentimentClassification(
        "uclanlp/visualbert-nlvr2-coco-pre",
        dropout_rate=0.5  # Increased dropout
    ).to(DEVICE)
    
    visual_model_pos = VisualBertForSentimentClassification(
        "uclanlp/visualbert-nlvr2-coco-pre",
        dropout_rate=0.5
    ).to(DEVICE)
    
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model_neg = CLIPForSentimentAnalysis(
        "openai/clip-vit-base-patch32",
        dropout_rate=0.5
    ).to(DEVICE)
    
    clip_model_pos = CLIPForSentimentAnalysis(
        "openai/clip-vit-base-patch32",
        dropout_rate=0.5
    ).to(DEVICE)
    
    feature_extractor = ResNetFeatureExtractor().to(DEVICE)
    
    # Initialize optimizers with different learning rates
    optimizer_visual_neg = AdamW(visual_model_neg.parameters(), lr=2e-5, weight_decay=0.01)
    optimizer_visual_pos = AdamW(visual_model_pos.parameters(), lr=2e-5, weight_decay=0.01)
    optimizer_clip_neg = AdamW(clip_model_neg.parameters(), lr=2e-5, weight_decay=0.01)
    optimizer_clip_pos = AdamW(clip_model_pos.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Use Focal Loss instead of BCE
    focal_loss_hate = FocalLoss(alpha=0.25, gamma=2.0).to(DEVICE)
    focal_loss_antihate = FocalLoss(alpha=0.25, gamma=2.0).to(DEVICE)
    
    # Create model manager with separate loss functions
    model_manager = ModelManager(
        visual_model_neg=visual_model_neg,
        visual_model_pos=visual_model_pos,
        clip_model_neg=clip_model_neg,
        clip_model_pos=clip_model_pos,
        feature_extractor=feature_extractor,
        tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
        loss_fn = focal_loss_hate,
        clip_tokenizer=clip_tokenizer,
        optimizer_visual_neg=optimizer_visual_neg,
        optimizer_visual_pos=optimizer_visual_pos,
        optimizer_clip_neg=optimizer_clip_neg,
        optimizer_clip_pos=optimizer_clip_pos,
        smoothing=0.05  # Reduced label smoothing
    )

    # Phase 1: Train on balanced dataset
    print("\nStarting Phase 1: Training on balanced dataset...")
    model_manager.train_phase1(train_loader, val_loader, num_epochs_phase1)
    
    # Phase 2: Fine-tune on minority classes
    print("\nStarting Phase 2: Fine-tuning on minority classes...")
    model_manager.train_phase2(
        hate_minority_loader, 
        antihate_minority_loader, 
        val_loader, 
        num_epochs_phase2
    )
    
    # Generate training progress plots
    print("\nGenerating training progress plots...")
    model_manager.plot_training_progress()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results, test_metrics = model_manager.evaluate_test_set(test_loader)
    
    # Save results
    results_df = pd.DataFrame(test_results)
    results_df.to_csv('test_results.csv', index=False)
    
    with open('test_metrics.txt', 'w') as f:
        f.write("Test Set Metrics:\n")
        for model_name, model_metrics in test_metrics.items():
            f.write(f"\n{model_name}:\n")
            for metric_name, value in model_metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
    
    print("\nTraining and evaluation complete!")
    print("Results have been saved to 'test_results.csv' and 'test_metrics.txt'")
    
    return model_manager, test_results, test_metrics

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('/fs/cml-scratch/kseelman/Meme_Project/multi-modal/hate/merged_final_hate_train_df.csv')
    df = df.rename(columns={
        'img_path': 'image',
        'image_text': 'Image Text',
    })
    
    # Create dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = MyDataset(df, tokenizer, transform=transform)
    
    # Run two-phase training
    model_manager, test_results, test_metrics = train_two_phase(
        dataset,
        num_epochs_phase1=NUM_EPOCHS_PHASE1,
        num_epochs_phase2=NUM_EPOCHS_PHASE2
    )
