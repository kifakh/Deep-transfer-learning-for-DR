# Diabetic Retinopathy Disease Classification using Deep Learning

This repository contains MATLAB code for the classification of Diabetic Retinopathy (DR) disease using deep learning techniques. The project utilizes transfer learning methods such as DenseNet, Inception, etc., and applies various techniques including 5-fold cross-validation, data augmentation, fine-tuning, and fusion using soft voting. The dataset used for experimentation is APTOS2019.

## Dataset

The APTOS2019 dataset is utilized for training and evaluation. It contains high-resolution retinal images labeled with five severity grades of diabetic retinopathy: 0 (no DR), 1 (mild), 2 (moderate), 3 (severe), and 4 (proliferative DR).

## Deep Learning Models

The following transfer learning models are implemented for classification:

- **DenseNet**: A densely connected convolutional network architecture known for its efficient feature reuse.
- **Inception**: A convolutional neural network architecture designed for better utilization of computational resources.
- **(Add more models as applicable)**

## Techniques Applied

The project employs the following techniques to enhance classification performance:

- **5-Fold Cross-Validation**: Data is split into five folds, and the model is trained and evaluated on each fold separately to assess its generalization performance.
- **Data Augmentation**: Techniques such as rotation, scaling, and flipping are applied to increase the diversity of the training data.
- **Fine-Tuning**: Pre-trained models are fine-tuned on the APTOS2019 dataset to adapt them to the specific task of DR classification.
- **Fusion using Soft Voting**: Predictions from multiple models are combined using soft voting to improve overall classification accuracy.

## Usage

1. Clone this repository to your local machine.
2. Download the APTOS2019 dataset and place it in the appropriate directory.
3. Open the MATLAB scripts for the desired deep learning model (e.g., `densenet_classification.m`, `inception_classification.m`).
4. Execute the script to train and evaluate the model on the APTOS2019 dataset.


## Conclusion

This project demonstrates the application of deep learning techniques for the classification of Diabetic Retinopathy disease using MATLAB. The use of transfer learning, data augmentation, fine-tuning, and fusion techniques contributes to improved classification accuracy and robustness.
