# Utility functions and factory methods for modular project

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

def get_model(model_name, **kwargs):
    if model_name == 'clip':
        from models.clip_model import CLIPForSentimentAnalysis
        return CLIPForSentimentAnalysis(**kwargs)
    elif model_name == 'visualbert':
        from models.visualbert_model import VisualBertForSentimentClassification
        return VisualBertForSentimentClassification(**kwargs)
    elif model_name == 'resnet':
        from models.resnet_feature_extractor import ResNetFeatureExtractor
        return ResNetFeatureExtractor(**kwargs)
    elif model_name == 'multitask_visualbert':
        from models.multitask_visualbert import MultiTaskVisualBERT
        return MultiTaskVisualBERT(**kwargs)
    elif model_name == 'multitask_clip':
        from models.multitask_clip import MultiTaskCLIP
        return MultiTaskCLIP(**kwargs)
    else:
        raise ValueError(f'Unknown model: {model_name}')

def get_loss(loss_name, **kwargs):
    if loss_name == 'cross_entropy':
        from losses.cross_entropy import get_loss
        return get_loss(**kwargs)
    elif loss_name == 'focal':
        from losses.focal_loss import get_loss
        return get_loss(**kwargs)
    elif loss_name == 'bce_with_logits':
        from losses.bce_with_logits import get_loss
        return get_loss(**kwargs)
    else:
        raise ValueError(f'Unknown loss: {loss_name}')

def get_dataset(dataset_name, **kwargs):
    if dataset_name == 'meme_hate':
        from datasets.meme_hate import MemeHateDataset
        return MemeHateDataset(**kwargs)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

def get_trainer(trainer_name, *args, **kwargs):
    if trainer_name == 'base':
        from trainers.base_trainer import BaseTrainer
        return BaseTrainer(*args, **kwargs)
    elif trainer_name == 'multitask':
        from trainers.multitask_trainer import MultiTaskTrainer
        return MultiTaskTrainer(*args, **kwargs)
    elif trainer_name == 'ensemble':
        from trainers.ensemble_trainer import EnsembleTrainer
        return EnsembleTrainer(*args, **kwargs)
    else:
        raise ValueError(f'Unknown trainer: {trainer_name}')

def preprocess_images_for_clip(images):
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    if isinstance(images, (list, tuple)):
        processed_images = [preprocess(image) if not torch.is_tensor(image) else image for image in images]
        return torch.stack(processed_images)
    elif isinstance(images, Image.Image):
        return preprocess(images).unsqueeze(0)
    else:
        return images

def evaluate_model(model, dataloader, device=None, hate_threshold=0.5, anti_hate_threshold=0.5, is_ensemble=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_hate_labels = []
    all_anti_hate_labels = []
    all_hate_preds = []
    all_anti_hate_preds = []
    all_hate_probs = []
    all_anti_hate_probs = []
    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch
            if is_ensemble:
                _, images, texts, inputs, hate_labels, anti_hate_labels, *_ = batch
                results = model(images, texts, inputs)
                hate_probs = results['hate_probs']
                anti_hate_probs = results['anti_hate_probs']
            else:
                # For single models, expect (img_url, image, text, inputs, negative_sentiment, positive_sentiment, topic_dist)
                _, images, _, inputs, hate_labels, anti_hate_labels, topic_dist = batch
                images = images.to(device)
                input_ids = inputs['input_ids'].squeeze(1).to(device)
                attention_mask = inputs['attention_mask'].squeeze(1).to(device)
                topic_dist = topic_dist.to(device) if topic_dist is not None else None
                hate_labels = hate_labels.to(device)
                anti_hate_labels = anti_hate_labels.to(device)
                # Try CLIP/VisualBERT forward signatures
                try:
                    hate_logits = model(input_ids, images, topic_dist)
                except Exception:
                    # Try VisualBERT signature
                    visual_embeds = model.feature_extractor(images)
                    if len(visual_embeds.shape) == 2:
                        visual_embeds = visual_embeds.unsqueeze(1)
                    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=device)
                    visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long, device=device)
                    hate_logits = model(input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids, topic_dist)
                hate_probs = sigmoid(hate_logits).squeeze()
                anti_hate_probs = hate_probs  # For single-task models, use same output
            hate_preds = (hate_probs >= hate_threshold).int()
            anti_hate_preds = (anti_hate_probs >= anti_hate_threshold).int()
            all_hate_labels.extend(hate_labels.cpu().numpy())
            all_anti_hate_labels.extend(anti_hate_labels.cpu().numpy())
            all_hate_preds.extend(hate_preds.cpu().numpy())
            all_anti_hate_preds.extend(anti_hate_preds.cpu().numpy())
            all_hate_probs.extend(hate_probs.cpu().numpy())
            all_anti_hate_probs.extend(anti_hate_probs.cpu().numpy())
    hate_metrics = calculate_metrics(all_hate_labels, all_hate_probs, all_hate_preds)
    anti_hate_metrics = calculate_metrics(all_anti_hate_labels, all_anti_hate_probs, all_anti_hate_preds)
    return {'hate': hate_metrics, 'anti_hate': anti_hate_metrics}

def calculate_metrics(actual, predicted_probs, predicted_labels=None, threshold=0.5):
    actual = np.array([a[0] if isinstance(a, (list, np.ndarray)) else a for a in actual])
    predicted_probs = np.array(predicted_probs)
    if predicted_labels is None:
        predicted_labels = (predicted_probs >= threshold).astype(int)
    else:
        predicted_labels = np.array(predicted_labels)
    accuracy = accuracy_score(actual, predicted_labels)
    precision = precision_score(actual, predicted_labels, zero_division=0)
    recall = recall_score(actual, predicted_labels, zero_division=0)
    f1 = f1_score(actual, predicted_labels, zero_division=0)
    try:
        auc = roc_auc_score(actual, predicted_probs)
    except Exception:
        auc = 0.5
    precision_weighted = precision_score(actual, predicted_labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(actual, predicted_labels, average='weighted', zero_division=0)
    f1_weighted = f1_score(actual, predicted_labels, average='weighted', zero_division=0)
    precision_macro = precision_score(actual, predicted_labels, average='macro', zero_division=0)
    recall_macro = recall_score(actual, predicted_labels, average='macro', zero_division=0)
    f1_macro = f1_score(actual, predicted_labels, average='macro', zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    } 