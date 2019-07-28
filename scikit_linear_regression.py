import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#defining function for linear regression
def linear_regression(x_train, y_train, x_test):
    linear_model = LinearRegression() 
    linear_model.fit(x_train,y_train)
    predictions = linear_model.predict(x_test)  
    return predictions    
	
if __name__ == '__main__':

    #reading csv files
    train_data = pd.read_csv('training_data.csv', header=None)
    train_label = pd.read_csv('training_labels.csv', header=None)	
    test_data = pd.read_csv('testing_data.csv', header=None)
    test_label = pd.read_csv('testing_labels.csv', header=None)

    #calling our defined linear regression function
    predicts = linear_regression(train_data, train_label, test_data);

    #printing prediction values
    print("Value of predictions for test dataset:")
    for y_values in np.nditer(predicts):
        print(y_values)

    #Calculating RSME
    y_test = test_label.values
    squared_sum = 0.0
    for index in range(len(predicts)):
        squared_sum = squared_sum + (predicts[index] - y_test[index])**2
    mean_squared_sum = squared_sum/len(predicts);
    rmse = math.sqrt(mean_squared_sum)
    
    print("\nValue of RMSE:", rmse)
