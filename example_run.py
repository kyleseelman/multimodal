import os
import yaml
import torch
from utils import get_model, get_loss, get_dataset, get_trainer, evaluate_model
from trainers.model_manager import ModelManager
from trainers.ensemble_trainer import EnsembleTrainer
from trainers.multitask_trainer import MultiTaskTrainer
from models.resnet_feature_extractor import ResNetFeatureExtractor
from transformers import AutoTokenizer, CLIPTokenizer
from interpretability import AttentionInterpreter
from PIL import Image

# 1. Load config
config_path = 'configs/all_models.yaml'  # Change as needed
with open(config_path) as f:
    config = yaml.safe_load(f)

# 2. Prepare tokenizers, dataset, dataloaders
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
clip_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')

dataset_args = config['dataset_args']
dataset_args['tokenizer'] = tokenizer
dataset = get_dataset(config['dataset'], **dataset_args)

from torch.utils.data import DataLoader, random_split
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# 3. Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = get_loss(config['loss'], **config.get('loss_args', {}))
model_dir = config.get('model_dir', 'saved_models')
os.makedirs(model_dir, exist_ok=True)

trainer_type = config['trainer']

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
    manager.train_and_validate(train_loader, val_loader, config['epochs'])
    torch.save(visual_model_neg.state_dict(), os.path.join(model_dir, 'visual_model_neg.pth'))
    torch.save(visual_model_pos.state_dict(), os.path.join(model_dir, 'visual_model_pos.pth'))
    torch.save(clip_model_neg.state_dict(), os.path.join(model_dir, 'clip_model_neg.pth'))
    torch.save(clip_model_pos.state_dict(), os.path.join(model_dir, 'clip_model_pos.pth'))

elif trainer_type == 'multitask':
    from models.multitask_visualbert import MultiTaskVisualBERT
    from models.multitask_clip import MultiTaskCLIP
    from multitask_learning import MultiTaskModelManager
    feature_extractor = ResNetFeatureExtractor().to(device)
    visual_model = get_model('multitask_visualbert', visual_bert_model_name=config['model_args']['visualbert_model_name'], dropout_rate=config['model_args']['dropout_rate'])
    clip_model = get_model('multitask_clip', clip_model_name=config['model_args']['clip_model_name'], dropout_rate=config['model_args']['dropout_rate'])
    optimizer_visual = torch.optim.AdamW(visual_model.parameters(), lr=config['learning_rate'])
    optimizer_clip = torch.optim.AdamW(clip_model.parameters(), lr=config['learning_rate'])
    multitask_manager = MultiTaskModelManager(
        visual_model, clip_model, feature_extractor, tokenizer, clip_tokenizer,
        loss_fn, loss_fn, optimizer_visual, optimizer_clip
    )
    multitask_manager.train_and_validate(train_loader, val_loader, config['epochs'])
    torch.save(visual_model.state_dict(), os.path.join(model_dir, 'multitask_visualbert.pth'))
    torch.save(clip_model.state_dict(), os.path.join(model_dir, 'multitask_clip.pth'))

elif trainer_type == 'ensemble':
    # See README for ensemble setup
    pass

else:
    model = get_model(config['model'], **config['model_args'])
    trainer = get_trainer(config['trainer'], model, loss_fn, dataset, config)
    trainer.train()
    torch.save(model.state_dict(), os.path.join(model_dir, f"{config['model']}_model.pth"))

# 4. Evaluation / Testing (example for single model)
model = get_model(config['model'], **config['model_args'])
model.load_state_dict(torch.load(os.path.join(model_dir, f"{config['model']}_model.pth")))
model.to(device)
metrics = evaluate_model(model, test_loader, device)
print(metrics)

# For multitask or all-models, load and evaluate each as in main.py

# 5. Interpretability / Attention Visualization
image = Image.open('path/to/image.jpg')
input_args = {
    'input_ids': ...,           # e.g., torch tensor for text
    'attention_mask': ...,     # if needed
    # ... any other model-specific inputs
}
interpreter = AttentionInterpreter(model)
# For gradcam, specify the target layer if needed
attn_map = interpreter.interpret(
    image,
    input_args,
    method='gradcam',
    target_layer=None,
    show=True
) 