# Historical Image Colorization

This project focuses on colorizing historical grayscale images using a deep learning approach based on the U-Net architecture. The model is trained to map grayscale images to their corresponding color images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)


## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/historical-image-colorization.git
cd historical-image-colorization
pip install -r requirements.txt
```

Alternatively, you can install the required libraries directly:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras opencv-python
```

## Usage

After installing the dependencies, you can run the notebook to train and evaluate the model. The dataset used is CIFAR-10, and the notebook includes all the necessary steps for data preprocessing, model training, and evaluation.

## Model Architecture

The colorization model is based on the U-Net architecture, which consists of an encoder to down-sample the images and a decoder to up-sample them back to the original size. The model is designed to learn the mapping from grayscale to color images.

## Training

The model is trained on the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes. During training, the images are converted to grayscale and normalized. The model is trained using mean squared error (MSE) as the loss function.

```python
history = model.fit(x_train_gray, x_train, epochs=100, batch_size=64, validation_data=(x_val_gray, x_val))
```

print(f'Test Loss: {loss}')
```

## Results

Include here any images or visualizations showing the results of the colorization process.
