import torch
from torch.utils.data import DataLoader

class MultiTaskTrainer:
    def __init__(self, visual_model, clip_model, feature_extractor, loss_fn_hate, loss_fn_anti_hate, dataset, config):
        self.visual_model = visual_model
        self.clip_model = clip_model
        self.feature_extractor = feature_extractor
        self.loss_fn_hate = loss_fn_hate
        self.loss_fn_anti_hate = loss_fn_anti_hate
        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_model.to(self.device)
        self.clip_model.to(self.device)
        self.feature_extractor.to(self.device)
        self.optimizer_visual = torch.optim.AdamW(self.visual_model.parameters(), lr=config.get('learning_rate', 1e-4))
        self.optimizer_clip = torch.optim.AdamW(self.clip_model.parameters(), lr=config.get('learning_rate', 1e-4))
        self.batch_size = config.get('batch_size', 8)
        self.epochs = config.get('epochs', 10)

    def train(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.visual_model.train()
        self.clip_model.train()
        for epoch in range(self.epochs):
            total_loss_visual, total_loss_clip = 0, 0
            for batch in dataloader:
                _, images, _, inputs, hate_sentiment, anti_hate_sentiment, topic_dist = batch
                images = images.to(self.device)
                input_ids = inputs['input_ids'].squeeze(1).to(self.device)
                attention_mask = inputs['attention_mask'].squeeze(1).to(self.device)
                topic_dist = topic_dist.to(self.device)
                hate_sentiment = hate_sentiment.to(self.device)
                anti_hate_sentiment = anti_hate_sentiment.to(self.device)
                # VisualBERT multitask
                visual_embeds = self.feature_extractor(images)
                if len(visual_embeds.shape) == 2:
                    visual_embeds = visual_embeds.unsqueeze(1)
                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=self.device)
                visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long, device=self.device)
                hate_logits_visual, anti_hate_logits_visual = self.visual_model(
                    input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids)
                loss_hate_visual = self.loss_fn_hate(hate_logits_visual, hate_sentiment)
                loss_anti_hate_visual = self.loss_fn_anti_hate(anti_hate_logits_visual, anti_hate_sentiment)
                loss_visual = loss_hate_visual + loss_anti_hate_visual
                self.optimizer_visual.zero_grad()
                loss_visual.backward()
                self.optimizer_visual.step()
                total_loss_visual += loss_visual.item()
                # CLIP multitask
                hate_logits_clip, anti_hate_logits_clip = self.clip_model(input_ids, images)
                loss_hate_clip = self.loss_fn_hate(hate_logits_clip, hate_sentiment)
                loss_anti_hate_clip = self.loss_fn_anti_hate(anti_hate_logits_clip, anti_hate_sentiment)
                loss_clip = loss_hate_clip + loss_anti_hate_clip
                self.optimizer_clip.zero_grad()
                loss_clip.backward()
                self.optimizer_clip.step()
                total_loss_clip += loss_clip.item()
            print(f"Epoch {epoch+1}/{self.epochs} - VisualBERT Loss: {total_loss_visual/len(dataloader):.4f} - CLIP Loss: {total_loss_clip/len(dataloader):.4f}") 