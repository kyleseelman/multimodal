import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class EnsembleTrainer:
    def __init__(self, ensemble_model, dataset, config):
        self.ensemble_model = ensemble_model
        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 10)
        self.loss_fn_hate = config.get('loss_fn_hate')
        self.loss_fn_anti_hate = config.get('loss_fn_anti_hate')
        self.ensemble_type = getattr(ensemble_model, 'ensemble_type', 'weighted')

    def train(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        if self.ensemble_type == 'learnable':
            self.ensemble_model.train_weights(dataloader, self.loss_fn_hate, self.loss_fn_anti_hate, num_epochs=self.epochs)
        else:
            print(f"No training required for ensemble type: {self.ensemble_type}")

    def evaluate(self, dataloader, hate_threshold=0.5, anti_hate_threshold=0.5):
        self.ensemble_model.eval()
        all_hate_labels = []
        all_anti_hate_labels = []
        all_hate_preds = []
        all_anti_hate_preds = []
        all_hate_probs = []
        all_anti_hate_probs = []
        with torch.no_grad():
            for _, images, texts, inputs, hate_labels, anti_hate_labels, *_ in dataloader:
                results = self.ensemble_model(images, texts, inputs)
                hate_probs = results['hate_probs']
                anti_hate_probs = results['anti_hate_probs']
                hate_preds = (hate_probs >= hate_threshold).int()
                anti_hate_preds = (anti_hate_probs >= anti_hate_threshold).int()
                all_hate_labels.extend(hate_labels.cpu().numpy())
                all_anti_hate_labels.extend(anti_hate_labels.cpu().numpy())
                all_hate_preds.extend(hate_preds.cpu().numpy())
                all_anti_hate_preds.extend(anti_hate_preds.cpu().numpy())
                all_hate_probs.extend(hate_probs.cpu().numpy())
                all_anti_hate_probs.extend(anti_hate_probs.cpu().numpy())
        hate_metrics = self._calculate_metrics(all_hate_labels, all_hate_probs, all_hate_preds)
        anti_hate_metrics = self._calculate_metrics(all_anti_hate_labels, all_anti_hate_probs, all_anti_hate_preds)
        return {'hate': hate_metrics, 'anti_hate': anti_hate_metrics}

    def _calculate_metrics(self, labels, probs, preds):
        labels = np.array([label[0] if isinstance(label, (list, np.ndarray)) else label for label in labels])
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
            'auc': roc_auc_score(labels, probs)
        }
        metrics.update({
            'precision_weighted': precision_score(labels, preds, average='weighted', zero_division=0),
            'recall_weighted': recall_score(labels, preds, average='weighted', zero_division=0),
            'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
            'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
            'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
            'f1_macro': f1_score(labels, preds, average='macro', zero_division=0)
        })
        return metrics 