# MNIST Image Classification with TensorFlow

This project leverages TensorFlow and TensorFlow Datasets to perform image classification on the MNIST dataset. The code includes data loading, normalization, and the creation of training and testing pipelines.

## Data Loading

The MNIST dataset is loaded into training and test sets, with additional information retrieved for analysis.

## Data Preprocessing

Images are normalized, converting pixel values from uint8 to float32. Training and test datasets undergo parallel processing, caching, shuffling, batching, and prefetching for optimal performance.

## Model Architecture and Training

A neural network model is created for image classification, with configurable parameters such as the number of epochs, learning rate, number of neurons, and activation function. The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss function.

## Hyperparameter Tuning

The script explores various hyperparameter combinations, including epochs, learning rates, number of neurons, and activation functions. Training is performed for each combination, and relevant metrics are stored for analysis.

## Best Model Selection

The script identifies the best models based on criteria such as minimal training loss, maximum training accuracy, minimal validation loss, and maximum validation accuracy.

## Results and Evaluation

The final results, including hyperparameters and evaluation metrics, are displayed for the best-performing models on both the training and test datasets.
Feel free to modify hyperparameter values and explore additional configurations to optimize the model further.
