# VisualBERT model definition for modular use

import torch
import torch.nn as nn
from transformers import VisualBertModel

class VisualBertForSentimentClassification(nn.Module):
    def __init__(self, visual_bert_model_name, dropout_rate=0.1, topic_dim=20):
        super().__init__()
        self.visual_bert = VisualBertModel.from_pretrained(visual_bert_model_name)
        self.classifier = nn.Linear(self.visual_bert.config.hidden_size + topic_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids, topic_dist=None):
        outputs = self.visual_bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   visual_embeds=visual_embeds,
                                   visual_attention_mask=visual_attention_mask,
                                   visual_token_type_ids=visual_token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        if topic_dist is not None:
            combined_features = torch.cat([pooled_output, topic_dist.float().to(pooled_output.device)], dim=1)
        else:
            combined_features = pooled_output
        logits = self.classifier(combined_features)
        return logits 