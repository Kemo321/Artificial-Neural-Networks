# Project 04: Traffic Sign Generation (DCGAN)

## Task Description
This project focuses on the generative domain of computer vision. The task is to build a generative framework capable of synthesizing realistic, high-quality images of traffic signs across multiple distinct structural classes.

## Technical Approach
* **Framework:** Deep Convolutional Generative Adversarial Networks (**DCGAN**).
* **Architecture Details:**
  * **Generator:** Utilizes a sequence of Fractional-Strided Convolutions (`nn.ConvTranspose2d`), Batch Normalization, and `ReLU` activations, culminating in a `Tanh` output layer to project latent noise vectors $z \in \mathbb{R}^{100}$ into RGB images.
  * **Discriminator:** Formulated as a deep convolutional classifier using `nn.Conv2d`, **Spectral Normalization** (to stabilize adversarial training gradients), and `LeakyReLU` activations.
* **Training Dynamics:** Modeled as a minimax game utilizing Binary Cross-Entropy Loss (`nn.BCELoss`) with independent Adam optimizers for both sub-networks.
* **Evaluation:** Image quality and distribution alignment were quantitatively evaluated using the **Fréchet Inception Distance (FID)** metric.