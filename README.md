# Multi-Modal Neural Network for Digit Classification

This repository contains the code and report for a **Multi-Modal Neural Network Model** designed to classify handwritten digits using **image** and **audio** data streams. The project demonstrates the power of multi-modal data fusion in improving classification performance.

## Overview

The project tackles the challenge of digit classification by combining image and audio data using a fusion-based deep learning approach. Separate convolutional neural networks (CNNs) were used to encode image and audio data, and their representations were fused for classification. The results showed significant improvements compared to single-modality models.

## Key Features

- **Multi-Modal Data Fusion**: Combines features from image and audio inputs for robust classification.
- **CNN-Based Encoding**: Uses convolutional neural networks to extract features from image and audio data.
- **High Performance**: Achieves an F1 Score of **0.978** on the Kaggle test dataset.
- **Extensive Analysis**: Includes clustering and embedding visualizations to demonstrate model performance.

## Repository Contents

- **Report**: A detailed write-up explaining the methodology, model architecture, training process, results, and future work recommendations ([PDF file](Report-Multimodal-NN-Digit-Classification.pdf)).
- **Code**: A Jupyter Notebook implementing the model, data preprocessing, training, and evaluation ([Notebook file](Multi-Modal-NN_Notebook.ipynb)).

## Methodology

1. **Data Preprocessing**: 
   - Images: 28x28 grayscale pixels.
   - Audio: Normalized 1D arrays of length 507.
   - Data batched and prepared for efficient training.

2. **Model Design**:
   - Separate CNN-based encoders for image and audio.
   - Fused encodings passed through a fully connected neural network for classification.

3. **Training**:
   - Optimized using MSE loss for encoders and categorical cross-entropy for the fusion model.
   - Used Adam optimizer with hyperparameter tuning.

4. **Evaluation**:
   - Metrics: Accuracy and F1 Score.
   - Visualization of embeddings and clustering to assess feature separation.

## Results

- **Fusion Model**: F1 Score of **0.978**, outperforming individual models.
- **Single-Modality Models**:
  - Image-only: Accuracy **95.1%**
  - Audio-only: Accuracy **83.4%**

## Future Work

- Experimenting with advanced architectures like transformers.
- Exploring automated hyperparameter optimization.
- Extending the fusion model for more complex tasks.

## References

This work draws inspiration from studies on multi-modal learning and neural network architectures, with references provided in the report.

## License

[MIT License](LICENSE)
