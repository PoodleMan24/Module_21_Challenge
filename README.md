# Module_21_Challenge
Baileys UWA Submission for Module 21 challenge
This Challenge required use of Tensor Flow to create a neural network for Alphabet Soups csv file


## Overview of the Analysis
The purpose of this analysis was to develop and evaluate a deep learning model to predict the success of loans based on various input features. The goal was to create a neural network model that can effectively classify loans as either successful or unsuccessful. This involved several stages including data preprocessing, model design, compilation, training, and evaluation.

# Results
Data Preprocessing

Target Variable(s):

* IS_SUCCESSFUL: This binary variable indicates whether a loan was successful (1) or not (0). It is the target variable we aim to predict.
Feature Variables:

* All variables except IS_SUCCESSFUL were considered features. This includes categorical variables converted to dummy variables, and numerical variables such as ASK_AMT, INCOME_AMT, etc.
Variables to Remove:

* Columns that were neither features nor the target variable were removed. This includes identifiers or columns that do not provide useful information for the model.
### Compiling, Training, and Evaluating the Model

Neural Network Architecture:

* Neurons and Layers:
    * Input Layer: 128 neurons
    * First Hidden Layer: 128 neurons with ReLU activation
    * Second Hidden Layer: 64 neurons with ReLU activation
    * Third Hidden Layer: 32 neurons with ReLU activation
    * Output Layer: 1 neuron with sigmoid activation (for binary classification)
Reasoning: The choice of neurons and layers was made to balance model complexity and performance. More neurons and layers allow the model to capture more complex patterns but also increase the risk of overfitting.
Model Performance:

Target Performance: The target model performance was an accuracy of 0.75 or higher.
Achieved Performance:
The model achieved an accuracy of 0.73 on the test set, which is slightly below the target but demonstrates a good level of performance.
Steps to Increase Performance:

Model Complexity: Increased the number of neurons and layers to improve the model’s ability to learn complex patterns.
Regularization: Added dropout layers to prevent overfitting.
Optimization: Used the Adam optimizer with a learning rate of 0.001.
Early Stopping: Implemented early stopping to prevent overfitting and to save the best model.

Summary
The deep learning model showed promising results with an accuracy close to the target performance of 0.75. The model’s complexity was adjusted with additional layers and neurons, and regularization techniques like dropout were employed to prevent overfitting. Despite the slightly lower accuracy than the target, the model demonstrates good predictive performance.

Recommendation:
To further enhance performance, consider experimenting with different architectures, such as:

Ensemble Methods: Combining multiple models (e.g., Random Forests, Gradient Boosting) could provide better accuracy.
Hyperparameter Tuning: Use techniques like grid search or random search to find optimal hyperparameters.
Alternative Models: Explore other algorithms like Support Vector Machines or XGBoost for potentially better results.
In conclusion, while the current neural network model performs well, additional experimentation and model tuning are recommended to achieve higher accuracy and better classification results.