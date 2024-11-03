# Emotion Prediction Using Text Classification with Neural Networks

## Project Overview

This project aims to classify text data into different emotion categories using a neural network model. The dataset consists of text samples labeled with emotions, and the model leverages embedding layers and dense layers to predict the emotion conveyed by a given text.

## Features

- **Text Preprocessing**: Tokenizes and pads the text data to ensure consistency in input length.
- **Label Encoding**: Converts emotion labels into one-hot encoded format suitable for multi-class classification.
- **Neural Network Architecture**: Uses an embedding layer, a flatten layer, and fully connected layers to classify emotions.
- **Training and Evaluation**: Trains the model with categorical cross-entropy and evaluates its performance on a test set.

## Dataset

- **Text Emotion Dataset** (`train.txt`): Contains text samples and corresponding emotion labels, with columns `Text` (text data) and `Emotions` (emotion labels).

## Requirements

- **Python 3.10**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Keras**
- **TensorFlow**

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/Tanjim-Islam/Emotion-Prediction-Using-Text-Classification-with-Neural-Networks.git
    cd Emotion-Prediction-Using-Text-Classification-with-Neural-Networks
    ```

2. **Install Dependencies:**

    ```bash
    pip install pandas numpy scikit-learn keras tensorflow
    ```

3. **Ensure the Dataset is Available:**
   Place `train.txt` in the working directory.

## Code Structure

1. **Data Loading and Preparation**:
   - Loads the dataset and renames columns as `Text` and `Emotions`.
   - Tokenizes and pads text data.
   - Encodes labels into integer values, followed by one-hot encoding for classification.

2. **Model Building**:
   - Defines a sequential neural network with:
     - An `Embedding` layer to capture text representations.
     - A `Flatten` layer to reshape data.
     - Dense layers, including a softmax output layer for multi-class classification.

3. **Training and Evaluation**:
   - Compiles the model with categorical cross-entropy and the Adam optimizer.
   - Trains the model on the training data with validation against a test set.

4. **Example Prediction**:
   - Prepares and tokenizes a sample input text for prediction.
   - Outputs the predicted emotion label.