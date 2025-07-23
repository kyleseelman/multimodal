import torch
from torch.utils.data import DataLoader

class BaseTrainer:
    def __init__(self, model, loss_fn, dataset, config):
        self.model = model
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.get('learning_rate', 1e-4))
        self.batch_size = config.get('batch_size', 8)
        self.epochs = config.get('epochs', 10)

    def train(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                *_, inputs, labels, _, topic_dist = batch
                inputs = {k: v.squeeze(1).to(self.device) for k, v in inputs.items()}
                topic_dist = topic_dist.to(self.device) if topic_dist is not None else None
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(**inputs, topic_dist=topic_dist)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss/len(dataloader):.4f}") 