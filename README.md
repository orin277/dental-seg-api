# 🦷 API for neural network models for tooth and caries segmentation

This project provides an API for neural network models that perform tooth and caries segmentation on panoramic dental X-ray images.
The API is built with FastAPI and integrates PyTorch models for deep learning inference.

## 🚀 Features

- Tooth Segmentation Endpoint – uses an ensemble of 3 models (Dense U-Net, Attention U-Net, and U-Net 3+) to predict a tooth mask.
- Caries Segmentation Endpoint – first applies the tooth segmentation ensemble to locate the teeth region, crops the relevant area, and then passes it to another ensemble (Dense U-Net, Attention U-Net, and U-Net 3+) to predict caries regions.
- Healthcheck Endpoint – simple endpoint to verify the API’s operational status.

Each prediction returns the path to the saved segmentation mask.

## 🧠 Technologies Used

- FastAPI – for building a high-performance REST API
- PyTorch – for implementing and running neural networks
- Albumentations – for image preprocessing and augmentations
- TorchMetrics – for model evaluation metrics
- OpenCV – for image processing operations
- Pytest – for automated testing
- Logging – for runtime tracking and debugging