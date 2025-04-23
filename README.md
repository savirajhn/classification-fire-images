# Fire/Non-Fire Image Classification

This project aims to build a Convolutional Neural Network (CNN) model to classify images as either containing fire or not containing fire (non-fire).

## Project Overview

The project follows these key steps:

1. **Data Collection:** Gather a dataset of images labeled as "fire" or "non-fire."
2. **Data Preprocessing:** Resize images, normalize pixel values, and split the dataset into training, validation, and testing sets.
3. **Model Building:** Construct a CNN model using TensorFlow/Keras with convolutional, pooling, and dense layers.
4. **Model Training:** Train the model using the training data and monitor its performance on the validation set.
5. **Model Evaluation:** Evaluate the trained model's accuracy on the testing set.
6. **Model Conversion (TFJS):** Convert the trained Keras model to TensorFlow.js format for use in web applications.

## Dataset

The dataset used for this project consists of images of fire and non-fire scenes. The dataset was split into:
- Training set: 64% of the data
- Validation set: 16% of the training data (used to monitor the model's performance during training)
- Testing set: 20% of the data (used to evaluate the final model's performance)

## Model Architecture

The CNN model architecture comprises the following layers:

- Input layer: Accepts images of size 256x256 pixels with 3 color channels.
- Convolutional layers: Extract features from the input images using filters with ReLU activation.
- Max pooling layers: Reduce the spatial dimensions of feature maps by downsampling.
- Dense layers: Process the extracted features to make predictions.
- Dropout layer: Prevent overfitting by randomly dropping out neurons during training.
- Output layer: Produces the final classification with 2 neurons representing fire and non-fire (softmax activation).

## Training and Evaluation

The model was trained using the Adam optimizer and categorical cross-entropy loss function. The training process was monitored with accuracy metrics.

To prevent overfitting, the `Dropout` layer and early stopping were employed. The final model achieved a test accuracy of [insert your test accuracy here]%.


## TensorFlow.js Conversion

The trained Keras model was converted to the TensorFlow.js format using the `tensorflowjs` library. This allows the model to be used for inference in web browsers or other JavaScript environments.

## Usage

To use the model for inference:
1. Load the model in a web application or Node.js environment.
2. Preprocess the input image.
3. Pass the image through the model.
4. Interpret the output probabilities to determine the classification (fire/non-fire).
## Dependencies

- TensorFlow 2.x
- Keras
- TensorFlow.js
- scikit-learn (for data splitting)
- pandas (for data management)
- Pillow (for image loading)
- pathlib (for file handling)