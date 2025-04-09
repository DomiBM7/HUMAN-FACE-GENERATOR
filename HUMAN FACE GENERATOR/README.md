# Facial Image Generation using GANs

## Author
BUKASA MUYOMBO

## Project Overview
This project implements a Generative Adversarial Network (GAN) for generating facial images using TensorFlow/Keras. The GAN consists of a generator and discriminator network that work together to produce realistic facial images from random noise.

## Features
- Custom GAN implementation using TensorFlow/Keras
- Generator network with transpose convolutional layers
- Discriminator network with convolutional layers
- Real-time training monitoring with image generation
- Label smoothing for better training stability
- Adaptive learning rates for both networks

## Architecture

### Discriminator
- Input shape: (64, 64, 3) - RGB images
- Multiple convolutional layers with increasing filters (64 to 512)
- LeakyReLU activation (alpha=0.2)
- Dropout layer (0.2) for regularization
- Binary classification output (real/fake)

### Generator
- Input: Random noise vector (latent_dimensions=128)
- Dense layer and reshape operations
- Multiple transpose convolutional layers
- LeakyReLU activation layers
- Final tanh activation for image generation
- Output shape: (64, 64, 3) - RGB images

## Training Details
- Epochs: 100
- Batch Size: 40
- Learning Rate:
  - Discriminator: 0.0002
  - Generator: 0.0001
- Loss Function: Binary Cross Entropy
- Image Size: 64x64 pixels
- Dataset: celebA dataset

## Requirements
- See requirements.txt for detailed dependencies
- NVIDIA GPU recommended for faster training

## Usage
1. Prepare the celebA dataset in the project directory
2. Install required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the training script:
```bash
python gan_training.py
```

## Output
- Generated images are saved as PNG files
- Naming format: `image_[epoch]_[index].png`
- 10 sample images are generated after each epoch

## Monitoring
The training process can be monitored through:
- Discriminator loss (d_loss)
- Generator loss (g_loss)
- Generated image samples after each epoch