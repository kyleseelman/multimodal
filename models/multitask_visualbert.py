import torch
import torch.nn as nn
from transformers import VisualBertModel

class MultiTaskVisualBERT(nn.Module):
    """Multi-task VisualBERT that jointly predicts hate and anti-hate"""
    def __init__(self, visual_bert_model_name, dropout_rate=0.1):
        super().__init__()
        self.visual_bert = VisualBertModel.from_pretrained(visual_bert_model_name)
        self.shared_layer = nn.Linear(self.visual_bert.config.hidden_size, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.hate_classifier = nn.Linear(256, 1)
        self.anti_hate_classifier = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids):
        outputs = self.visual_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids
        )
        pooled_output = outputs.pooler_output
        shared_features = self.shared_layer(pooled_output)
        shared_features = self.dropout(shared_features)
        hate_logits = self.hate_classifier(shared_features)
        anti_hate_logits = self.anti_hate_classifier(shared_features)
        return hate_logits, anti_hate_logits 