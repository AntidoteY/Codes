#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 03:17:59 2024

@author: Yiyao Li, Ocean Yu
"""
import pandas as pd
import CRApkg as CRA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import tensorflow as tf
import time
from tensorflow.keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")

'''Step 1'''
start_time = time.time()
# Use CRA.py to create the bank data
data = CRA.load_BankData()
'''Step 2'''
# Put default into a "y" variable. Choose your x variables and put them into a numpy array. 
y = data.default
x = data[['equity_assets', 'TCOs_loans', 'nonII_assets']].copy()

'''Step 3'''
X_train, X_val, Y_train, Y_val = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)
# Early stopping callback
early_stopping = EarlyStopping(monitor='auc', mode='max', verbose=0, patience=20,
                                      restore_best_weights=True)

# performance metrics
def performStats(model_in):
    PD_val = model_in.predict(X_val).flatten()
    PD_train = model_in.predict(X_train).flatten()
    loglike_val = np.sum(np.log(PD_val) * Y_val + np.log(1 - PD_val) * (1 - Y_val))
    AUC_val = roc_auc_score(Y_val, PD_val)
    AR_val = 2 * (AUC_val - .5)

    loglike_train = np.sum(np.log(PD_train) * Y_train + np.log(1 - PD_train) * (1 - Y_train))
    AUC_train = roc_auc_score(Y_train, PD_train)
    AR_train = 2 * (AUC_train - .5)
    
    parms = model_in.count_params()
    AIC_val = 2 * parms - 2 * loglike_val
  
    result = {
        'AUC_val': AUC_val, 'AR_val': AR_val, "ll_val": loglike_val, "AIC_val": AIC_val,
        'AUC_train': AUC_train, 'AR_train': AR_train, "ll_train": loglike_train, "parms": parms,
        'ops_train': len(X_train), "obs_val": len(X_val)
    }
    return result


# create, compile, and train a model
def train_model(hidden_nodes, output_activation, optimizer, loss_fn):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_nodes, activation='sigmoid', use_bias=True),
        tf.keras.layers.Dense(1, activation=output_activation, use_bias=True)
    ])
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['AUC'])
    model.fit(X_train, Y_train, epochs=50, verbose=1, validation_data=(X_val, Y_val), callbacks=[early_stopping])
    return model

# List of configurations to test
models = [
    (5, 'sigmoid', 'adam', 'binary_crossentropy'),
    (5, 'relu', 'adam', 'binary_crossentropy'),
    (2, 'sigmoid', 'adam', 'binary_crossentropy'),
    (2, 'relu', 'adam', 'binary_crossentropy'),
    (5, 'sigmoid', 'sgd', 'binary_crossentropy'),
    (5, 'relu', 'sgd', 'binary_crossentropy'),
    (2, 'sigmoid', 'sgd', 'binary_crossentropy'),
    (2, 'relu', 'sgd', 'binary_crossentropy'),
    (5, 'sigmoid', 'rmsprop', 'binary_crossentropy'),
    (5, 'relu', 'rmsprop', 'binary_crossentropy'),
    (2, 'sigmoid', 'rmsprop', 'binary_crossentropy'),
    (2, 'relu', 'rmsprop', 'binary_crossentropy'),
    (5, 'sigmoid', 'adagrad', 'binary_crossentropy'),
    (5, 'relu', 'adagrad', 'binary_crossentropy'),
    (2, 'sigmoid', 'adagrad', 'binary_crossentropy'),
    (2, 'relu', 'adagrad', 'binary_crossentropy'),
    (32, 'sigmoid', 'adam', 'binary_crossentropy' ),
    (32, 'relu', 'adam', 'binary_crossentropy'),
    (32, 'sigmoid', 'sgd', 'binary_crossentropy'),
    (32,'relu', 'sgd', 'binary_crossentropy'),
    (32, 'sigmoid', 'rmsprop', 'binary_crossentropy'),
    (32,'relu', 'rmsprop', 'binary_crossentropy'),
    (32, 'sigmoid', 'adagrad', 'binary_crossentropy'),
    (32,'relu', 'adagrad', 'binary_crossentropy'),
]

models_performance = {}

# Train models and store their performance
for (hidden_nodes, output_activation, optimizer, loss_fn) in models:
    model = train_model(hidden_nodes, output_activation, optimizer, loss_fn)
    model_name = f"{hidden_nodes} Hidden Sigmoid, Output {output_activation.capitalize()}, {optimizer.capitalize()}"
    performance_metrics = performStats(model)
    models_performance[model_name] = performance_metrics
    print(f"Performance ({model_name}):", performance_metrics)

# Sorting and printing the models based on AUC for validation data
sorted_models_auc = sorted(models_performance.items(), key=lambda x: x[1]['AUC_val'], reverse=True)
print("Sorted Models Based on Validation AUC:")
for model_name, metrics in sorted_models_auc:
    print(f"Model: {model_name}")
    print(f"AUC (Validation): {metrics['AUC_val']}")
    print(f"AIC (Validation): {metrics['AIC_val']}")
    print(f"AR (Validation): {metrics['AR_val']}")
    print(f"Log-Likelihood (Validation): {metrics['ll_val']}")
    print("-----" * 10)
 
''' print best 5 models '''  
best_models = sorted_models_auc[:5]
print("\nTop 5 Models Based on Validation AUC:")
for model_name, metrics in best_models:
    print(f"Model: {model_name}")
    print(f"AUC (Validation): {metrics['AUC_val']}")
    print(f"AIC (Validation): {metrics['AIC_val']}")
    print(f"AR (Validation): {metrics['AR_val']}")
    print(f"Log-Likelihood (Validation): {metrics['ll_val']}")
    print("-----" * 10)
''' 
based on the result, we find that the 32 hidden nodes model's AUC is higher than
the 5 hidden nodes model which has same input and output activation. the logliklihood and AR are higher. 
as the same number of hidden nodes, and same activation. the Adam optimizer perform
better and stable. 

'''
    
  
''' Step 4: a table of the in-sample (the training sample) results'''
# Extract metrics for in-sample (training) and out-of-sample (validation) results
def create_results_table(models_performance, sample_type='validation'):
    rows = []
    for model_name, metrics in models_performance.items():
        if sample_type == 'validation':
            row = {
                'Model': model_name,
                'Observations': metrics['obs_val'],
                'Defaults': sum(Y_val),
                'AUC': metrics['AUC_val'],
                'AR': metrics['AR_val'],
                'Log-Likelihood': metrics['ll_val'],
                'Parameters': metrics['parms'],
                'AIC': metrics['AIC_val']
            }
        else:  # training sample
            row = {
                'Model': model_name,
                'Observations': metrics['ops_train'],
                'Defaults': sum(Y_train),
                'AUC': metrics['AUC_train'],
                'AR': metrics['AR_train'],
                'Log-Likelihood': metrics['ll_train'],
                'Parameters': metrics['parms'],
                'AIC': 2 * metrics['parms'] - 2 * metrics['ll_train']
            }
        rows.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Calculate AIC weights
    min_aic = df['AIC'].min()
    df['AIC Weight'] = np.exp(-0.5 * (df['AIC'] - min_aic)) / np.exp(-0.5 * (df['AIC'] - min_aic)).sum()
    
    return df

# Create the in-sample results table
train_results_table = create_results_table(models_performance, sample_type='training')
print("\nIn-Sample (Training) Results:")
print(train_results_table)

# Create the out-of-sample results table
validation_results_table = create_results_table(models_performance, sample_type='validation')
print("\nOut-of-Sample (Validation) Results:")
print(validation_results_table)


'''
In-sample results:
Among models with the same hidden nodes and output activation,
using the Adam optimizer generally have much higher AUC and AR than those using the SGD optimizer, 
particularly with the sigmoid output activation;
Models with linear output activation and the RMSprop optimizer show strong performance, 
particularly in terms of AR and AUC;
The Adam optimizer tends to perform well across different configurations of 
hidden nodes and output activations, indicating good fit on the training data.

Out-of-sample results:
Similarly, in the validation set, models using the Adam optimizer continue to o
utperform those using the SGD optimizer, especially with sigmoid output activation.
The Adam optimizer generally achieves the highest AUC and AR, 
suggesting it provides a good balance between model complexity and generalization ability.
The Adam optimizer consistently yields better performance across various architectures and activation
in both in-sample and out-of-sample evaluations, making it the preferred choice among the tested options.

Models like "2 Hidden Sigmoid, Output Sigmoid, Adam" 
have the highest Akaike weights, suggesting they have the best balance of fit and complexity in the validation set;

Models with very low Akaike weights are less likely to be the best model, 
suggesting they either overfit the training data or do not generalize well.

'''


'''
Essay:

The superior performance of models using the Adam optimizer across various configurations. 
Adam's adaptive learning rate mechanism proved beneficial, allowing the network to converge 
more quickly and reliably compared to other optimizers like SGD and RMSprop. 
The stability provided by Adam was evident in the higher AUC and AR values, 
indicating a better balance between convergence speed and accuracy.

Exploring different node counts revealed that models with a moderate number of hidden nodes tended to perform best. 
Too few nodes led to underfitting, where the model struggled to capture the complexity of the data. 
Conversely, too many nodes increased the risk of overfitting, particularly noticeable 
when the difference between in-sample and out-sample performance widened.

The choice of activation functions also played a crucial role. 
Sigmoid activations, while useful for binary classification problems, 
sometimes led to saturation issues, making it harder for the network to learn. 
In contrast, linear activations in the output layer showed promise, 
particularly when combined with RMSprop, offering stable and high-performing models in both in-sample and out-of-sample evaluations.

The relatively modest impact of node counts on AUC and AR also underscores 
the importance of other hyperparameters, such as the choice of activation function and optimization algorithm.
different optimizer and activations will impact the result a lot.
'''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {end_time - start_time:.2f} seconds")