# Artificial Neural Networks - Academic Projects

This repository contains a series of projects developed during the Artificial Neural Networks course. The curriculum covers a wide range of deep learning tasks, from tabular data regression to advanced generative computer vision models.

## Projects Overview

### [01. Bike Rental Prediction](./01-Bike-Rental-Prediction/)
* **Type:** Regression (MLP)
* **Goal:** Predict the number of bike rentals ($cnt$) based on weather and seasonal data.
* **Metric:** Root Mean Squared Logarithmic Error (RMSLE).

### [02. Apartment Price Classifier](./02-Apartment-Price-Classifier/)
* **Type:** Classification (Tabular)
* **Goal:** Classify apartments into price categories (Cheap, Average, Expensive) based on a \$350,000 budget.
* **Challenge:** Handling highly imbalanced classes.

### [03. Image Classification CNN](./03-Image-Classification-CNN/)
* **Type:** Computer Vision (Classification)
* **Goal:** 50-class image classifier built from scratch.
* **Constraint:** No pre-trained models allowed; custom CNN architecture design.

### [04. Traffic Sign Generation](./04-Traffic-Sign-Generation/)
* **Type:** Computer Vision (Generative)
* **Goal:** Generate synthetic traffic signs using GANs.
* **Metric:** Evaluated using Frechet Inception Distance (FID).

### [05. Classical Composer RNN](./05-Classical-Composer-RNN/)
* **Type:** Sequence Classification (RNN/LSTM/GRU)
* **Goal:** Predict the composer of a classical piece based on its sequence of chords.
* **Challenge:** Handling variable-length sequences using padding and sequence packing in PyTorch.

### [06. Hate Speech Classification](./06-Hate-Speech-Classification/)
* **Type:** NLP (Text Classification)
* **Goal:** Detect hate/offensive speech in Polish-language comments.
* **Model:** HerBERT (`allegro/herbert-base-cased`) fine-tuned for sequence classification.
* **Pipeline:** Tokenization (`HerbertTokenizer`), transformer fine-tuning (Cross-Entropy), inference on unlabeled test data.

---
## Tech Stack
* **Language:** Python
* **Library:** PyTorch, Torchvision
* **Tools:** NumPy, Pandas, Matplotlib, Scikit-learn