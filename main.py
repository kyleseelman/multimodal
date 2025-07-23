import argparse
import yaml
from utils import get_model, get_loss, get_dataset, get_trainer, evaluate_model
from trainers.model_manager import ModelManager
from trainers.ensemble_trainer import EnsembleTrainer
from trainers.multitask_trainer import MultiTaskTrainer
import torch
from transformers import AutoTokenizer, CLIPTokenizer
from models.resnet_feature_extractor import ResNetFeatureExtractor
import os

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--test', action='store_true', help='Run evaluation on test set')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    trainer_type = config['trainer']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer setup (replace with your tokenizer loading logic)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    # Dataset
    dataset_args = config['dataset_args']
    dataset_args['tokenizer'] = tokenizer
    dataset = get_dataset(config['dataset'], **dataset_args)

    # Loss
    loss_fn = get_loss(config['loss'], **config.get('loss_args', {}))

    model_dir = config.get('model_dir', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)

    from torch.utils.data import DataLoader, random_split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    if trainer_type == 'model_manager':
        visual_model_neg = get_model('visualbert', visual_bert_model_name=config['model_args']['visualbert_model_name'], dropout_rate=config['model_args']['dropout_rate'], topic_dim=config['model_args']['topic_dim'])
        visual_model_pos = get_model('visualbert', visual_bert_model_name=config['model_args']['visualbert_model_name'], dropout_rate=config['model_args']['dropout_rate'], topic_dim=config['model_args']['topic_dim'])
        clip_model_neg = get_model('clip', clip_model_name=config['model_args']['clip_model_name'], dropout_rate=config['model_args']['dropout_rate'], topic_dim=config['model_args']['topic_dim'])
        clip_model_pos = get_model('clip', clip_model_name=config['model_args']['clip_model_name'], dropout_rate=config['model_args']['dropout_rate'], topic_dim=config['model_args']['topic_dim'])
        feature_extractor = ResNetFeatureExtractor().to(device)
        optimizer_visual_neg = torch.optim.AdamW(visual_model_neg.parameters(), lr=config['learning_rate'])
        optimizer_visual_pos = torch.optim.AdamW(visual_model_pos.parameters(), lr=config['learning_rate'])
        optimizer_clip_neg = torch.optim.AdamW(clip_model_neg.parameters(), lr=config['learning_rate'])
        optimizer_clip_pos = torch.optim.AdamW(clip_model_pos.parameters(), lr=config['learning_rate'])
        manager = ModelManager(
            visual_model_neg, visual_model_pos, clip_model_neg, clip_model_pos,
            feature_extractor, tokenizer, clip_tokenizer, loss_fn,
            optimizer_visual_neg, optimizer_visual_pos, optimizer_clip_neg, optimizer_clip_pos
        )
        if not args.test:
            manager.train_and_validate(train_loader, val_loader, config['epochs'])
            save_model(visual_model_neg, os.path.join(model_dir, 'visual_model_neg.pth'))
            save_model(visual_model_pos, os.path.join(model_dir, 'visual_model_pos.pth'))
            save_model(clip_model_neg, os.path.join(model_dir, 'clip_model_neg.pth'))
            save_model(clip_model_pos, os.path.join(model_dir, 'clip_model_pos.pth'))
        else:
            load_model(visual_model_neg, os.path.join(model_dir, 'visual_model_neg.pth'), device)
            load_model(visual_model_pos, os.path.join(model_dir, 'visual_model_pos.pth'), device)
            load_model(clip_model_neg, os.path.join(model_dir, 'clip_model_neg.pth'), device)
            load_model(clip_model_pos, os.path.join(model_dir, 'clip_model_pos.pth'), device)
            print('Evaluating VisualBERT (hate)...')
            metrics_vb_neg = evaluate_model(visual_model_neg, test_loader, device)
            print(metrics_vb_neg)
            print('Evaluating VisualBERT (anti-hate)...')
            metrics_vb_pos = evaluate_model(visual_model_pos, test_loader, device)
            print(metrics_vb_pos)
            print('Evaluating CLIP (hate)...')
            metrics_clip_neg = evaluate_model(clip_model_neg, test_loader, device)
            print(metrics_clip_neg)
            print('Evaluating CLIP (anti-hate)...')
            metrics_clip_pos = evaluate_model(clip_model_pos, test_loader, device)
            print(metrics_clip_pos)
    elif trainer_type == 'ensemble':
        # Assume pretrained models are available
        visual_model_neg = get_model('visualbert', visual_bert_model_name=config['model_args']['visualbert_model_name'], dropout_rate=config['model_args']['dropout_rate'], topic_dim=config['model_args']['topic_dim'])
        visual_model_pos = get_model('visualbert', visual_bert_model_name=config['model_args']['visualbert_model_name'], dropout_rate=config['model_args']['dropout_rate'], topic_dim=config['model_args']['topic_dim'])
        clip_model_neg = get_model('clip', clip_model_name=config['model_args']['clip_model_name'], dropout_rate=config['model_args']['dropout_rate'], topic_dim=config['model_args']['topic_dim'])
        clip_model_pos = get_model('clip', clip_model_name=config['model_args']['clip_model_name'], dropout_rate=config['model_args']['dropout_rate'], topic_dim=config['model_args']['topic_dim'])
        # Load pretrained weights
        load_model(visual_model_neg, os.path.join(model_dir, 'visual_model_neg.pth'), device)
        load_model(visual_model_pos, os.path.join(model_dir, 'visual_model_pos.pth'), device)
        load_model(clip_model_neg, os.path.join(model_dir, 'clip_model_neg.pth'), device)
        load_model(clip_model_pos, os.path.join(model_dir, 'clip_model_pos.pth'), device)
        feature_extractor = ResNetFeatureExtractor().to(device)
        # Instantiate ensemble model (user should implement this class as needed)
        from trainers.ensemble_trainer import EnsembleTrainer
        from models.visualbert_model import VisualBertForSentimentClassification
        from models.clip_model import CLIPForSentimentAnalysis
        class SimpleEnsemble:
            def __init__(self, visual_model_neg, visual_model_pos, clip_model_neg, clip_model_pos):
                self.visual_model_neg = visual_model_neg
                self.visual_model_pos = visual_model_pos
                self.clip_model_neg = clip_model_neg
                self.clip_model_pos = clip_model_pos
                self.sigmoid = torch.nn.Sigmoid()
            def eval(self):
                self.visual_model_neg.eval()
                self.visual_model_pos.eval()
                self.clip_model_neg.eval()
                self.clip_model_pos.eval()
            def __call__(self, images, texts, inputs):
                # Example: average ensemble
                with torch.no_grad():
                    input_ids = inputs['input_ids'].squeeze(1).to(device)
                    attention_mask = inputs['attention_mask'].squeeze(1).to(device)
                    topic_dist = None
                    logits_vb_neg = self.visual_model_neg(input_ids, attention_mask, None, None, None, topic_dist)
                    logits_vb_pos = self.visual_model_pos(input_ids, attention_mask, None, None, None, topic_dist)
                    logits_clip_neg = self.clip_model_neg(input_ids, images, topic_dist)
                    logits_clip_pos = self.clip_model_pos(input_ids, images, topic_dist)
                    probs_vb_neg = self.sigmoid(logits_vb_neg)
                    probs_vb_pos = self.sigmoid(logits_vb_pos)
                    probs_clip_neg = self.sigmoid(logits_clip_neg)
                    probs_clip_pos = self.sigmoid(logits_clip_pos)
                    hate_probs = (probs_vb_neg + probs_clip_neg) / 2
                    anti_hate_probs = (probs_vb_pos + probs_clip_pos) / 2
                    return {'hate_probs': hate_probs, 'anti_hate_probs': anti_hate_probs}
        ensemble_model = SimpleEnsemble(visual_model_neg, visual_model_pos, clip_model_neg, clip_model_pos)
        trainer = EnsembleTrainer(ensemble_model, dataset, config)
        if not args.test:
            trainer.train()
            # Optionally save ensemble weights if learnable
        else:
            print('Evaluating Ensemble...')
            metrics = trainer.evaluate(test_loader)
            print(metrics)
    elif trainer_type == 'multitask':
        from trainers.multitask_trainer import MultiTaskTrainer
        from models.multitask_visualbert import MultiTaskVisualBERT
        from models.multitask_clip import MultiTaskCLIP
        from multitask_learning import MultiTaskModelManager
        feature_extractor = ResNetFeatureExtractor().to(device)
        visual_model = get_model('multitask_visualbert', visual_bert_model_name=config['model_args']['visualbert_model_name'], dropout_rate=config['model_args']['dropout_rate'])
        clip_model = get_model('multitask_clip', clip_model_name=config['model_args']['clip_model_name'], dropout_rate=config['model_args']['dropout_rate'])
        optimizer_visual = torch.optim.AdamW(visual_model.parameters(), lr=config['learning_rate'])
        optimizer_clip = torch.optim.AdamW(clip_model.parameters(), lr=config['learning_rate'])
        # Use MultiTaskModelManager for multitask training and evaluation
        if not args.test:
            multitask_manager = MultiTaskModelManager(
                visual_model, clip_model, feature_extractor, tokenizer, clip_tokenizer,
                loss_fn, loss_fn, optimizer_visual, optimizer_clip
            )
            multitask_manager.train_and_validate(train_loader, val_loader, config['epochs'])
            save_model(visual_model, os.path.join(model_dir, 'multitask_visualbert.pth'))
            save_model(clip_model, os.path.join(model_dir, 'multitask_clip.pth'))
        else:
            multitask_manager = MultiTaskModelManager(
                visual_model, clip_model, feature_extractor, tokenizer, clip_tokenizer,
                loss_fn, loss_fn, optimizer_visual, optimizer_clip
            )
            load_model(visual_model, os.path.join(model_dir, 'multitask_visualbert.pth'), device)
            load_model(clip_model, os.path.join(model_dir, 'multitask_clip.pth'), device)
            print('Evaluating multitask models...')
            data, metrics = multitask_manager.evaluate_test_set(test_loader)
            print(metrics)
    else:
        model = get_model(config['model'], **config['model_args'])
        trainer = get_trainer(config['trainer'], model, loss_fn, dataset, config)
        if not args.test:
            trainer.train()
            save_model(model, os.path.join(model_dir, f"{config['model']}_model.pth"))
        else:
            load_model(model, os.path.join(model_dir, f"{config['model']}_model.pth"), device)
            print(f'Evaluating {config["model"]}...')
            metrics = evaluate_model(model, test_loader, device)
            print(metrics)

if __name__ == '__main__':
    main() 