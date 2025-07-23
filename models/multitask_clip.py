import torch
import torch.nn as nn
from transformers import CLIPModel

class MultiTaskCLIP(nn.Module):
    """Multi-task CLIP that jointly predicts hate and anti-hate"""
    def __init__(self, clip_model_name, dropout_rate=0.1):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.shared_layer = nn.Linear(1, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.hate_classifier = nn.Linear(64, 1)
        self.anti_hate_classifier = nn.Linear(64, 1)

    def forward(self, input_ids, pixel_values):
        outputs = self.clip_model(input_ids=input_ids, pixel_values=pixel_values)
        logits = outputs.logits_per_image.diagonal().unsqueeze(1)
        shared_features = self.shared_layer(logits)
        shared_features = self.dropout(shared_features)
        hate_logits = self.hate_classifier(shared_features)
        anti_hate_logits = self.anti_hate_classifier(shared_features)
        return hate_logits, anti_hate_logits 