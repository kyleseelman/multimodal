import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

class ModelManager:
    def __init__(self, visual_model_neg, visual_model_pos, clip_model_neg, clip_model_pos, feature_extractor, tokenizer, clip_tokenizer, loss_fn, optimizer_visual_neg, optimizer_visual_pos, optimizer_clip_neg, optimizer_clip_pos, smoothing=0.1):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def smooth_labels(self, labels, smoothing=0.1):
        return labels * (1 - smoothing) + (smoothing / 2)

    def train(self, dataloader, sentiment_type, epoch_number):
        if sentiment_type == 'hate':
            visual_model = self.visual_model_neg
            clip_model = self.clip_model_neg
            optimizer_visual = self.optimizer_visual_neg
            optimizer_clip = self.optimizer_clip_neg
        else:
            visual_model = self.visual_model_pos
            clip_model = self.clip_model_pos
            optimizer_visual = self.optimizer_visual_pos
            optimizer_clip = self.optimizer_clip_pos
        visual_model.train()
        clip_model.train()
        for batch in dataloader:
            _, images, texts, inputs, negative_sentiments, positive_sentiments, topic_dist = batch
            images = images.to(self.device)
            input_ids = inputs['input_ids'].squeeze(1).to(self.device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(self.device)
            topic_dist = topic_dist.to(self.device)
            labels = (negative_sentiments if sentiment_type == 'hate' else positive_sentiments).float().to(self.device)
            labels = self.smooth_labels(labels, self.smoothing)
            optimizer_visual.zero_grad()
            optimizer_clip.zero_grad()
            with torch.no_grad():
                visual_embeds = self.feature_extractor(images)
            if len(visual_embeds.shape) == 2:
                visual_embeds = visual_embeds.unsqueeze(1)
            with autocast():
                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=self.device)
                visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long, device=self.device)
                logits_visual = visual_model(input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids, topic_dist)
                loss_visual = self.loss_fn(logits_visual, labels)
                text_inputs = self.clip_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                logits_clip = clip_model(input_ids=text_inputs.input_ids, pixel_values=images, topic_dist=topic_dist)
                loss_clip = self.loss_fn(logits_clip, labels)
            self.scaler.scale(loss_visual).backward()
            self.scaler.scale(loss_clip).backward()
            self.scaler.step(optimizer_visual)
            self.scaler.step(optimizer_clip)
            self.scaler.update()

    def validate(self, dataloader, sentiment_type):
        if sentiment_type == 'hate':
            visual_model = self.visual_model_neg
            clip_model = self.clip_model_neg
        else:
            visual_model = self.visual_model_pos
            clip_model = self.clip_model_pos
        visual_model.eval()
        clip_model.eval()
        self.feature_extractor.eval()
        total_loss_visual, total_loss_clip = 0, 0
        with torch.no_grad():
            for batch in dataloader:
                _, images, texts, inputs, negative_sentiments, positive_sentiments, topic_dist = batch
                images = images.to(self.device)
                input_ids = inputs['input_ids'].squeeze(1).to(self.device)
                attention_mask = inputs['attention_mask'].squeeze(1).to(self.device)
                topic_dist = topic_dist.to(self.device)
                labels = (negative_sentiments if sentiment_type == 'hate' else positive_sentiments).float().to(self.device)
                labels = self.smooth_labels(labels, self.smoothing)
                visual_embeds = self.feature_extractor(images)
                if len(visual_embeds.shape) == 2:
                    visual_embeds = visual_embeds.unsqueeze(1)
                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=self.device)
                visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long, device=self.device)
                logits_visual = visual_model(input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids, topic_dist)
                loss_visual = self.loss_fn(logits_visual, labels)
                text_inputs = self.clip_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                logits_clip = clip_model(input_ids=text_inputs.input_ids, pixel_values=images, topic_dist=topic_dist)
                loss_clip = self.loss_fn(logits_clip, labels)
                total_loss_visual += loss_visual.item()
                total_loss_clip += loss_clip.item()
        avg_loss_visual = total_loss_visual / len(dataloader)
        avg_loss_clip = total_loss_clip / len(dataloader)
        return avg_loss_visual, avg_loss_clip

    def train_and_validate(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.train(train_loader, 'hate', epoch)
            self.train(train_loader, 'anti_hate', epoch)
            val_loss_visual_neg, val_loss_clip_neg = self.validate(val_loader, 'hate')
            val_loss_visual_pos, val_loss_clip_pos = self.validate(val_loader, 'anti_hate')
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Hate - VisualBERT Loss: {val_loss_visual_neg:.4f}, CLIP Loss: {val_loss_clip_neg:.4f}")
            print(f"  Anti-Hate - VisualBERT Loss: {val_loss_visual_pos:.4f}, CLIP Loss: {val_loss_clip_pos:.4f}") 