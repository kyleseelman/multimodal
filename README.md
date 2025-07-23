# Meme Hate Classification Project

This project provides a modular framework for training CLIP, VisualBERT, and related models on meme hate classification tasks. Users can flexibly select models, loss functions, datasets, and training techniques via configuration files or command-line arguments.

## Project Structure

- `models/` — Model definitions (CLIP, VisualBERT, ResNet, etc.)
- `losses/` — Loss function modules (cross-entropy, focal loss, BCEWithLogits, etc.)
- `datasets/` — Dataset loading and preprocessing
- `trainers/` — Training logic (base, multitask, ensemble, model manager, etc.)
- `configs/` — Example YAML config files for experiments
- `main.py` — Unified entry point for training and evaluation
- `utils.py` — Factory functions, preprocessing, and evaluation helpers

## Installation

1. **Install dependencies**
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

---
