import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
from PIL import Image

class AttentionInterpreter:
    """
    Modular interpretability class for attention-based visualization.
    Supports any model with attention or gradient-based saliency.
    """
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        self.model.eval()

    def interpret(self, image, input_args, method='gradcam', target_layer=None, show=True, save_path=None):
        """
        Args:
            image: torch.Tensor (C, H, W) or PIL.Image
            input_args: dict of other model inputs (e.g., text, attention_mask, etc.)
            method: 'gradcam' (default), 'attention', or 'vanilla_grad'
            target_layer: layer to visualize (for gradcam/attention)
            show: whether to display the result
            save_path: if provided, saves the visualization
        Returns:
            The attention/saliency map as a numpy array.
        """
        if isinstance(image, Image.Image):
            image_tensor = torch.unsqueeze(self.preprocess(image), 0).to(self.device)
        else:
            image_tensor = image.unsqueeze(0).to(self.device)
        # Merge image into input_args
        model_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_args.items()}
        model_inputs['pixel_values'] = image_tensor
        # Forward pass to get output and attention
        if method == 'attention':
            # Try to extract attention weights from the model (if available)
            with torch.no_grad():
                outputs = self.model(**model_inputs, output_attentions=True)
                if hasattr(outputs, 'attentions'):
                    attn = outputs.attentions[-1].mean(1).squeeze().cpu().numpy()  # Last layer, mean heads
                else:
                    raise ValueError('Model does not return attention weights.')
            attn_map = self._resize_attention(attn, image_tensor.shape[-2:])
        elif method == 'gradcam':
            attn_map = self._gradcam(image_tensor, model_inputs, target_layer)
        elif method == 'vanilla_grad':
            attn_map = self._vanilla_gradient(image_tensor, model_inputs)
        else:
            raise ValueError(f'Unknown interpretability method: {method}')
        # Visualize
        if show or save_path:
            self._visualize(image_tensor, attn_map, show, save_path)
        return attn_map

    def preprocess(self, image):
        # Default preprocessing for CLIP/ViT
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
        preprocess = Compose([
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        return preprocess(image)

    def _resize_attention(self, attn, shape):
        # Resize attention map to image size
        from scipy.ndimage import zoom
        if attn.shape[-2:] != shape:
            zoom_factors = (shape[0] / attn.shape[-2], shape[1] / attn.shape[-1])
            attn = zoom(attn, zoom_factors, order=1)
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        return attn

    def _gradcam(self, image_tensor, model_inputs, target_layer):
        # Simple Grad-CAM for last conv layer
        activations = []
        gradients = []
        def forward_hook(module, inp, out):
            activations.append(out)
        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])
        handle = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_backward_hook(backward_hook)
        output = self.model(**model_inputs)
        score = output.logits.max() if hasattr(output, 'logits') else output.max()
        self.model.zero_grad()
        score.backward(retain_graph=True)
        act = activations[0].detach()
        grad = gradients[0].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        handle.remove()
        handle_b.remove()
        return cam

    def _vanilla_gradient(self, image_tensor, model_inputs):
        image_tensor.requires_grad_()
        output = self.model(**model_inputs)
        score = output.logits.max() if hasattr(output, 'logits') else output.max()
        self.model.zero_grad()
        score.backward()
        grad = image_tensor.grad.data.abs().squeeze().cpu().numpy()
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        return grad

    def _visualize(self, image_tensor, attn_map, show, save_path):
        img = to_pil_image(image_tensor.squeeze().cpu())
        attn_map = np.array(attn_map)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_map = np.uint8(255 * attn_map)
        attn_map = Image.fromarray(attn_map).resize(img.size, resample=Image.BILINEAR)
        attn_map = np.array(attn_map)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.imshow(attn_map, cmap='jet', alpha=0.5)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close() 