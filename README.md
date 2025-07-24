# Modular Meme Hate Classification Project

This project provides a modular framework for training CLIP, VisualBERT, and related models on meme hate classification tasks. Users can flexibly select models, loss functions, datasets, and training techniques via configuration files or command-line arguments.

## Project Structure

- `models/` — Model definitions (CLIP, VisualBERT, ResNet, etc.)
- `losses/` — Loss function modules (cross-entropy, focal loss, BCEWithLogits, etc.)
- `datasets/` — Dataset loading and preprocessing
- `trainers/` — Training logic (base, multitask, ensemble, model manager, etc.)
- `configs/` — Example YAML config files for experiments
- `main.py` — Unified entry point for training and evaluation
- `utils.py` — Factory functions, preprocessing, and evaluation helpers
- `interpretability.py` — Modular attention-based interpretability for any model

## Installation

1. **Install dependencies** (see `requirements.txt`, not included here)
2. **Prepare your dataset** and update the config file accordingly.

## Usage

### Training

- **Train all four models (VisualBERT-hate, VisualBERT-anti-hate, CLIP-hate, CLIP-anti-hate):**
  ```bash
  python main.py --config configs/all_models.yaml
  ```
- **Train only CLIP:**
  ```bash
  python main.py --config configs/clip_only.yaml
  ```
- **Train only VisualBERT:**
  ```bash
  python main.py --config configs/visualbert_only.yaml
  ```
- **Train ensemble (after training all four models):**
  ```bash
  python main.py --config configs/ensemble.yaml
  ```
- **Train multitask (joint CLIP and VisualBERT):**
  ```bash
  python main.py --config configs/multitask.yaml
  ```

### Evaluation / Testing

- **Evaluate all four models:**
  ```bash
  python main.py --config configs/all_models.yaml --test
  ```
- **Evaluate only CLIP:**
  ```bash
  python main.py --config configs/clip_only.yaml --test
  ```
- **Evaluate only VisualBERT:**
  ```bash
  python main.py --config configs/visualbert_only.yaml --test
  ```
- **Evaluate ensemble:**
  ```bash
  python main.py --config configs/ensemble.yaml --test
  ```
- **Evaluate multitask:**
  ```bash
  python main.py --config configs/multitask.yaml --test
  ```

## Interpretability / Attention Visualization

You can visualize what parts of the image your model is attending to using the modular interpretability module:

```python
from interpretability import AttentionInterpreter
from PIL import Image

# Load your trained model (e.g., CLIP, VisualBERT, multitask, etc.)
model = ...  # Load or instantiate your model
interpreter = AttentionInterpreter(model)

# Prepare your image and model inputs
image = Image.open('path/to/image.jpg')
input_args = {
    'input_ids': ...,           # e.g., torch tensor for text
    'attention_mask': ...,     # if needed
    # ... any other model-specific inputs
}

# For Grad-CAM, specify the target layer (e.g., last conv layer for CLIP/ResNet)
# target_layer = model.clip_model.visual.transformer.resblocks[-1].attn
# For ViT/CLIP, you may need to inspect your model for the correct layer

# Visualize attention (Grad-CAM, attention, or vanilla gradient)
attn_map = interpreter.interpret(
    image,
    input_args,
    method='gradcam',          # or 'attention', 'vanilla_grad'
    target_layer=None,         # Set to your model's target layer for gradcam
    show=True,                 # Show the visualization
    save_path=None             # Or provide a path to save
)
```

- **Supported methods:** `'gradcam'`, `'attention'`, `'vanilla_grad'`
- **Works with any model** as long as you provide the correct inputs and (for gradcam) the target layer.

## Configuration

Edit or create a config file in `configs/` to specify:
- `trainer`: Which trainer to use (`model_manager`, `base`, `ensemble`, `multitask`)
- `model`/`model_args`: Model type and parameters
- `loss`/`loss_args`: Loss function and parameters
- `dataset`/`dataset_args`: Dataset and preprocessing options
- `batch_size`, `epochs`, `learning_rate`, etc.

See the provided YAML files in `configs/` for templates.

## Extending
- Add new models to `models/` and update `utils.py`.
- Add new loss functions to `losses/` and update `utils.py`.
- Add new trainers or datasets similarly.

## Example Configs
- `configs/all_models.yaml`: Train all four models
- `configs/clip_only.yaml`: Train only CLIP
- `configs/visualbert_only.yaml`: Train only VisualBERT
- `configs/ensemble.yaml`: Ensemble training/evaluation
- `configs/multitask.yaml`: Multitask training/evaluation

