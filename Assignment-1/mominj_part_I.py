import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random

# We go about part1 by first reading in the csv, querying the year to get the needed data, and then doing some renaming and dropping null values 
# Next, we start splitting up the data into two sets, X and Y, which represent the feature and target vectors (both one column)
# We now initialize an instance of the LinearRegression class and give it the X and Y vectors
# We then preprocess the data by normalizing it and column stacking it into np arrays
# Now we declare the alpha values as our learning rates and epochs as our iterations and a dictionary to keep track of our betas
# We then cross each alpha with all the epochs and calculate their resulting betas using gradient descent
# After that, we get the betas from the OLS method and print out all of our betas (OLS and GD)
# For each beta, we now calculate the predicted target values and store them
# Finally, we start plotting the data by scattering the values and making two subplots, one with different beta values to show different regressions
# and one to show the OLS compared to the best beta (calculated through MSE)

# set random seed as professor requested
np.random.seed(42)

class LinearRegression:
    def __init__(self,X, Y):
        # when initializing the linear regression model, save the feature matrix and target vector
        self.X = X
        self.Y = Y
    
    def preprocess(self):
        # normalize all the data using the mean and standard deviation
        x_mean = np.mean(self.X)
        x_std = np.std(self.X)
        x_set = (self.X - x_mean) / x_std

        self.X = np.column_stack((np.ones(len(x_set)),x_set))
        
        y_mean = np.mean(self.Y)
        y_std = np.std(self.Y)
        y_set = (self.Y - y_mean) / y_std

        self.Y = np.column_stack((y_set)).T
        return None

    def ols(self):
        # return the ols betas calculated using the method shown in class
        return np.linalg.inv((self.X).T.dot(self.X)).dot((self.X).T).dot(self.Y)
    
    def gradient_descent(self,alpha,epoch):
        # initialize an array of random beta values
        beta = np.random.randn(2,1)
        
        # do the gradient descent algorithm iterating over it for the number of epochs and using the learning rate provided
        for _ in range(epoch):
            gradients = 2/len(self.X) * ((self.X).T).dot((self.X).dot(beta) - self.Y)
            beta = beta - alpha * gradients
        
        # return the calculated GD beta
        return beta
    
    def predict(self,beta):
        # multiply the beta vector and the feature matrix and sum up across the columns (so the row) to get the predicted Y values 
        return np.sum((self.X).dot(beta),axis=1)

def plot_indexed_betas(ax,X,Y,Y_hat,plot_col,beta_keys,x_label,y_label):
    # send in the subplot, the feature matrix, target vector, the predicted values, column number in the subplot, the key of the predicted value (related to the beta), and the labels
    # scatter the data
    ax[plot_col].scatter(X,Y)
    
    # for each beta that we want to print out, iterate over the corresponding key that was sent in
    for key in beta_keys:
        # plot the x feature vector and the predicted values with the key as the label
        ax[plot_col].plot(X,Y_hat[key][0],label=Y_hat[key][1])
    
    # set the labels and add the legend
    ax[plot_col].set_xlabel(x_label)
    ax[plot_col].set_ylabel(y_label)
    ax[plot_col].legend()
    return None

def get_best_beta(lin_reg_model,beta_dict):
    # take in the regression model to get the X and Y values, and the dictionary of betas
    X = lin_reg_model.X
    Y = lin_reg_model.Y
    
    # set the minimum mse as infinity and create an array to store the best beta
    min_mse = float('inf')
    best_beta = []
    
    # for each alpha and epoch key and beta values, you get the mse using the function by sending in the actual y values and predicted y values (X dot our beta)
    for key,beta in beta_dict.items():
        mse = get_mse(Y,X.dot(beta))
        
        # compare new mse to min mse, if it is smaller, set it as the new minimum and keep track of the best beta as an array of the value and the alpha epoch key
        if mse < min_mse:
            min_mse = mse
            best_beta = [beta,key]
    
    # return the best beta
    return best_beta

def get_mse(y,y_prime):
    # take in the actual y values and the predicted y values and get the length of it and create a variable to track the running sum of the mse
    n = len(y)
    summed_diff = 0
    
    # iterate over each value of y and y prime and get the squared error
    for i in range(n):
        summed_diff += (y[i] - y_prime[i])**2
        
    # finally take the mean and return it
    summed_diff /= n
    return summed_diff

# read in the csv and filter out extra rows that don't have year = 2018 using query
df = pd.read_csv('gdp-vs-happiness.csv')
df = df.query('Year == 2018')

# isolate the two rows that we need (happiness and gdp) and I decided to rename to make it easier to work wiht
df = df[['Cantril ladder score',"GDP per capita, PPP (constant 2017 international $)"]]
df.rename(columns={'Cantril ladder score' : 'Score', "GDP per capita, PPP (constant 2017 international $)" : 'GDP'}, inplace=True)

# had a few null entries like the ones for Africa so drop those rows
df = df.dropna()

# separate X feature vector and Y target vector from the df just by selecting those columns and turn into np arrays to make computation easier
X = np.array(df['GDP'])
Y = np.array(df['Score'])

# create an instance of my linear regression class with the inputs being the X feature matrix and Y target matrix and preprocess using the class function
lin_reg_model = LinearRegression(X,Y)
lin_reg_model.preprocess()

# set my alpha and epoch values and create a dictionary to track the betas that I am creating
alphas = [0.001,0.01,0.02,0.05,0.1]
epochs = [100,500,1000,1500,2000]
betas = dict()

# iterate over each alpha and each epoch, calculating the GD betas using a linear regression class function
for alpha in alphas:
    for epoch in epochs:
        betas[(alpha,epoch)] = lin_reg_model.gradient_descent(alpha,epoch)

# use the linear regression class function to get the OLS betas 
ols_betas = lin_reg_model.ols()

# print out all the betas generated by gradient descent
print("Beta's generated by gradient descent:\n")
for key,beta in betas.items():
    print(f'alpha: {key[0]}\t\tepochs: {key[1]}\t\tbeta0: {beta[0]}\t\t\tbeta1: {beta[1]}')

# print out the betas generated by OLS 
print("\nBeta's generated by OLS:\n")
print(f'\t\t\t\t\t\tbeta0:{ols_betas[0]}\t\t\tbeta1:{ols_betas[1]}')

# create a dictionary to store all the predicted y values with their alpha and epochs (of the betas) 
Y_hat = dict()
for key,beta in betas.items():
    Y_hat[key]=[lin_reg_model.predict(beta),str(key)]

# get all the values from the second column of X (ignoring the np.ones) and then flatten them to use for plotting, also get the Y target vector from the model
X_ = lin_reg_model.X[...,1].ravel()  
Y = lin_reg_model.Y

# creating two subplots for plotting normalized gdp vs happiness data
fig, ax = plt.subplots(1,2,figsize=(20,8))

# plot different GD betas on the first subplot (since I need to plot different alphas and epochs, I just manually choose which ones to plot that will not overlap, as requested by prof)
plot_indexed_betas(ax,X_,Y,Y_hat,0,[(0.001,500),(0.001,2000),(0.01,100),(0.001,1000),(0.05,500),(0.1,100),(0.1,500),(0.02,1500)],"GDP Per Capita","Happiness")

# get the best beta that was generated by GD and print it out (using mse)
best_beta = get_best_beta(lin_reg_model,betas)
print("\nBest beta generated by gradient descent (calculated using MSE):\n")
print(f'alpha:{best_beta[1][0]}\t\tepochs:{best_beta[1][1]}\t\tbeta0:{best_beta[0][0]}\t\t\tbeta1:{best_beta[0][1]}')

# create a dictionary to store the alpha and epochs of the best beta with the predicted values of it along with an entry to store the OLS and it's predictions
Y_prime = {best_beta[1] : [lin_reg_model.predict(best_beta[0]),str(best_beta[1])], "OLS" : [lin_reg_model.predict(ols_betas),"OLS"]}

# plot the predicted values from best beta and the predicted values from the OLS beta to see how they compare
plot_indexed_betas(ax,X_,Y,Y_prime,1,[(best_beta[1]),"OLS"],"GDP Per Capita", "Happiness")

# finally show the plots
plt.show()