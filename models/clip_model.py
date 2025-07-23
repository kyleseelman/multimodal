# CLIP model definition for modular use

import torch
import torch.nn as nn
from transformers import CLIPModel

class CLIPForSentimentAnalysis(nn.Module):
    def __init__(self, clip_model_name, dropout_rate=0.1, topic_dim=20):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1 + topic_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, pixel_values, topic_dist=None):
        outputs = self.clip_model(input_ids=input_ids, pixel_values=pixel_values)
        logits = outputs.logits_per_image.diagonal().unsqueeze(1)
        logits = self.dropout(logits)
        if topic_dist is not None:
            logits = torch.cat([logits, topic_dist.float().to(logits.device)], dim=1)
        logits = self.classifier(logits)
        return logits 