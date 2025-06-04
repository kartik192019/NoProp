# NoProp: Diffusion-based Label Propagation without Backpropagation

This project implements the NoProp training approach as described in the paper ["Procedural Content Generation via Generative Artificial Intelligence"](https://arxiv.org/abs/2407.09013). NoProp is a novel training method that uses diffusion-based label propagation without requiring traditional backpropagation between layers.

## Overview

NoProp introduces a new paradigm for training neural networks by:
- Using diffusion processes to propagate labels
- Training each layer independently
- Eliminating the need for end-to-end backpropagation

This implementation demonstrates NoProp on the MNIST digit classification task.

## Architecture

The model consists of:
1. A CNN feature extractor
2. Multiple MLPs (one per diffusion step)
3. A diffusion-based label propagation mechanism

Key components:
- `CNN`: Extracts features from input images
- `DenoisingMLP`: Denoises labels at each step
- Linear noise schedule from α=1.0 to α=0.1

## Requirements

```
torch
torchvision
matplotlib
numpy
```

## Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install torch torchvision matplotlib numpy
```

## Usage

The main notebook `NoProp.ipynb` contains:
1. Model implementation
2. Training loop
3. Inference code
4. Visualization utilities

To train the model:
```python
# Hyperparameters
T = 10  # Diffusion steps
embed_dim = 10  # Label embedding dimension
batch_size = 128
lr = 0.001
epochs = 50

# Train
for epoch in range(epochs):
    # Training loop implementation
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
```

## Results

The model achieves competitive accuracy on MNIST digit classification while offering several advantages:
- Parallel training of layers
- No backpropagation between layers
- Memory efficiency during training

Example prediction visualization is included in the notebook.

## Implementation Details

Key features:
- Linear noise schedule
- Independent MLPs per diffusion step
- CNN-based feature extraction
- MSE loss for label denoising
- Visualization tools for predictions

## Citation

If you use this code, please cite the original paper:
```
@article{mao2024procedural,
  title={Procedural Content Generation via Generative Artificial Intelligence},
  author={Mao, Xinyu and Yu, Wanli and Yamada, Kazunori D and Zielewski, Michael R},
  journal={arXiv preprint arXiv:2407.09013},
  year={2024}
}
```

## License

This project is released under the MIT License.
