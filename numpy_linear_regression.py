import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#function for calculating cost
def calculate_cost(x_train, y_train, theta):
    training_size = y_train.size
    predicted = np.dot(x_train, theta)
    sq_err = (predicted - y_train)
    cost = (1.0) / (2 * training_size) * np.dot(sq_err.T, sq_err)
    return cost

#function for gradient descent algorithm to find optimal theta values
def gradient_descent(x_train, y_train, theta, alpha, iterations):
    training_size = y_train.size
    theta_n = theta.size   
    theta_log = np.zeros(shape=(iterations+1, 1))    
    theta_log[0, 0] = calculate_cost(x_train, y_train, theta)
 
    for i in range(iterations):
        predicted = x_train.dot(theta)

        for thetas in range(theta_n):
            tmp = x_train[:,thetas]
            tmp.shape = (training_size ,1)
            err = (predicted - y_train) * tmp
            theta[thetas][0] = theta[thetas][0] - alpha * (1.0 / training_size) * err.sum()
        theta_log[i+1, 0] = calculate_cost(x_train, y_train, theta)

    return theta, theta_log

if __name__ == '__main__':

    #loading csv files
    train_data = pd.read_csv('training_data.csv', header=None)
    train_label = pd.read_csv('training_labels.csv', header=None)	
    test_data = pd.read_csv('testing_data.csv', header=None)
    test_label = pd.read_csv('testing_labels.csv', header=None)

    #converting frames to arrays
    x_train = train_data.values
    y_train = train_label.values
    x_test = test_data.values
    y_test = test_label.values

    m,n = np.shape(x_train)
    y_train.shape = (m, 1)
    #converting x_train to add x0
    x_train_converted = np.ones(shape=(m,1))
    x_train_converted = np.append(x_train_converted, x_train, 1)

    #set up initial thetas to 0
    theta = np.zeros(shape=(n+1, 1))
    #define number of iterations and alpha
    iterations = 200
    alpha = 0.1
    #calculate theta using gradient descent
    theta, J_theta_log = gradient_descent(x_train_converted, y_train, theta, alpha, iterations)

    #converting x_test to add x0
    x_test_converted = np.ones(shape=(m,1))
    x_test_converted = np.append(x_test_converted, x_test, 1)
    predictions = x_test_converted.dot(theta)

    #printing prediction values
    print("Value of predictions for test dataset:")
    for y_values in np.nditer(predictions):
        print(y_values)

    squared_sum = 0.0
    for index in range(m):
        squared_sum = squared_sum + (predictions[index] - y_test[index])**2    
    mean_squared_sum = squared_sum/m;
    rmse = math.sqrt(mean_squared_sum)
    print("\nValue of RMSE:", rmse)

