import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random

# We start part2 off by reading in the csv and using a lambda function to increase the number of rings by 1.5 to get the abalone age
# Then we do some renaming and split the data up into X and Y, a feature matrix and a target vector
# We now initialize an instance of the MultiLinearRegression class using X and Y
# Now, we set the number of folds and get all the k folds using it's function (function comments are in each function below)
# Next, we use a dictionary to store the result of the function which gets training and testing mse of each fold and it's created OLS betas
# From there, we get the average training and testing mse and get the OLS beta with the lowest testing mse
# We now normalize our original data and turn it into np array (did this here since it made getting my k folds easier, using pd instead of np)
# We then calculate our predicted target values using the beta we got from above
# To make graphing easier, we store each feature name and data in a dictionary which then gets passed to the plotting function
# Finally, we show all the predicted values plotting against each feature vector data on a scatter plot to see how it fits

# set random seed as professor requested
np.random.seed(42)

class MultiLinearRegression:
    def __init__(self,X,Y):
        # when initializing the linear regression model, save the feature matrix and target vector
        self.X = X
        self.Y = Y
    
    def preprocess(self):
        # for each feature vector in the feature matrix, normalize the data
        for col in self.X.columns:
            self.X[col] = (self.X[col] - np.mean(self.X[col])) / np.std(self.X[col])
        
        # normalize the target vector
        self.Y = (self.Y - np.mean(self.Y))/ np.std(self.Y)
        return None
    
    def ols(self,X,Y):
        # return the ols betas calculated using the method shown in class (using X and Y since I pass in different sets of data so easier than making multiple class instances)
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    def get_k_folds(self,k):
        # create an array to have the folds
        k_fold_list = []
        
        # get the size of each fold (size of the whole column divided by k)
        size = len(self.X) // k 
        
        # since i need to have k folds, i iterate over each fold index (like split number) of the k folds and do the needed row splitting
        for i in range(k):
            # |  0    |  ...  |   k-1    |       <- just to see what i mean by the fold index and index of where the fold starts (for the splicing)
            # i0     i1 .... ik-2    ik-1
            # break up the test set by splicing the matrix at the starting index of the fold to the next index of the fold (allows us to get the nth and n+1th fold index)
            # we do this for both feature matrix and target vector
            test_x = self.X.iloc[i*size:(i+1)*size,:] # since this is a matrix (more than one col), we need to specify all columns, Y is a vector so not needed
            test_y = self.Y.iloc[i*size:(i+1)*size]
            
            # since we want all the other entries besides the ones inside the test set, use those indices as the bounds for the new splicing
            # use concat to join them together getting us our training set, we do this for both X and Y
            train_x = pd.concat([self.X.iloc[:i*size,:],self.X.iloc[(i+1)*size:,:]],axis=0)
            train_y = pd.concat([self.Y.iloc[:i*size],self.Y.iloc[(i+1)*size:]],axis=0)
            
            # finally, make a test set and a train set and append it to the fold array
            k_fold_list.append([[test_x,test_y],[train_x,train_y]])
        return k_fold_list
    
    def get_train_test_mse(self,i,fold):
        # take in the fold and the iteration of the fold and split the fold into the corresponding training and testing sets
        test, train = fold[0],fold[1]
        test_x,test_y = test[0],test[1]
        train_x, train_y = train[0],train[1]
        
        # for the training set create a separate multi linear regression model instance using training x and y and preprocess it
        mlrm1 = MultiLinearRegression(train_x,train_y)
        mlrm1.preprocess()
        
        # get the preprocessed data from mlrm1 and column stack the matrices using np.ones for the x0 feature vector
        X_ = np.column_stack((np.ones(len(mlrm1.X)),mlrm1.X))
        Y_ = np.column_stack((mlrm1.Y)).T
        
        # doing the same thing as above but for the test set of data
        mlrm2 = MultiLinearRegression(test_x,test_y)
        mlrm2.preprocess()
        X_hat = np.column_stack((np.ones(len(mlrm2.X)),mlrm2.X))
        Y_hat = np.column_stack((mlrm2.Y)).T
        
        # call the ols function using the training dataset
        ols_betas = self.ols(X_,Y_)
        
        # print out the beta values using the print betas function
        print(f"Fold number {i} beta values are as follows:")
        print_betas(ols_betas)
        
        # get the training mse by getting the mse of the actual target values and the predicted values on training set
        train_mse = get_mse(Y_,X_.dot(ols_betas))
        
        # get the testing mse by getting the mse of the actual target values and the predicted values on testing set
        test_mse = get_mse(Y_hat,X_hat.dot(ols_betas))
        
        # print out the mses and return a list with all needed data
        print(f"For Fold number {i}, the train mse was {train_mse} and the test mse was {test_mse}\n")
        return [train_mse,test_mse,ols_betas]

def print_betas(beta_list):
    # for formatting print before and after to make space
    print()
    
    # go through each beta and print out it's number (b0, b1, etc) along with the beta value
    for i,beta in enumerate(beta_list):
        print(f"Beta{i}: {beta}")
    print()
    return None

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

def plot_feature_wise_subplots(feature_dict,Y,Y_prime,y_label,col):
    # initialize the i and j values (row and col) to track which subplot we are on
    i, j = 0,0
    
    # use a boolean flag so we stay in the first row for the first iteration
    first_go = True
    
    # for each key (column name) and feature data, we first scatter the data of that and the actual target values
    for key,feature in feature_dict.items():
        ax[i,j].scatter(feature,Y)
        
        # we then scatter the data based on the feature and the predicted target values and set the opacity less so we can see better how the model fits the data 
        ax[i,j].scatter(feature,Y_prime,color='r', alpha=0.3)
        
        # set out labels
        ax[i,j].set_xlabel(str(key))
        ax[i,j].set_ylabel(y_label)
        
        # go to the next column, if we wrap back around to 0 column, we switch to the next row (using a flag for the initial iteration so we can plot 0,0 without wrapping to 1,0)
        j = (j+1) % col
        if j == 0 and first_go==False:
            i += 1
        first_go = False
    return None

# read in the data from the csv
df = pd.read_csv("training_data.csv")

# use lambda function to increase ring number by 1.5 to get age and rename the column appropriately
df["Rings"] = df["Rings"].apply(lambda x : x + 1.5)
df.rename(columns={"Rings":"Age"},inplace=True)

# separate the target variable from the feature vector variables 
X = df.loc[:,df.columns!="Age"]
X.drop(columns=X.columns[0],axis=1,inplace=True)
Y = df["Age"]

# yi (Rings/Age) = b0 + b1 Length + b2 Diameter + b3 Height + b4 Whole_weight + b5 Shucked_weight + b6 Viscera_weight + b7 Shell_weight

# create an instance of the class multi linear regression and send in the feature matrix and the target vector
multi_lin_reg_model = MultiLinearRegression(X,Y)

# declare the number of folds you want for the k folds and get each fold
k = 5
folds = multi_lin_reg_model.get_k_folds(k)

# create a dictionary to store the train mses, test mses and betas
mse_dict = dict()

# for each fold, get the train mse test mse and betas
for i,fold in enumerate(folds):
    mse_dict[i] = multi_lin_reg_model.get_train_test_mse(i,fold)

# calculate the average train and test mse
train_avg = 0
test_avg = 0
for key,mses in mse_dict.items():
    train_avg += mses[0]
    test_avg += mses[1]
train_avg /= len(mse_dict)
test_avg /= len(mse_dict)
print(f"The training average mse was {train_avg} and the testing average mse was {test_avg}")

# calculate the best fold (so lowest test mse) and it's associated betas
my_beta = None
min_test_mse = float('inf')
best = 0
for key,mses in mse_dict.items():
    if mses[1] < min_test_mse:
        min_test_mse = mses[1]
        my_beta = mses[2]
        best = key

print(f"Best fold number was {best} with minimum test mse of {min_test_mse}")

# now preprocess the original data since we are done with the k folds cv and it won't affect anything else
multi_lin_reg_model.preprocess()

# get the newly normalized feature matrix data and column stack it with np.ones to get the x0 feature vector, column stack the target vector as well and transpose
X_ = np.column_stack((np.ones(len(multi_lin_reg_model.X)),multi_lin_reg_model.X))
Y = np.column_stack((multi_lin_reg_model.Y)).T

# calculate the predicted values for the whole dataset using the best calculated beta
Y_prime = np.sum(X_.dot(my_beta),axis=1) 

# create a dictionary to track hold each of the feature matrix's column data as a np array
X_feature_dict = dict()
for col in multi_lin_reg_model.X.columns:
    X_feature_dict[col] = multi_lin_reg_model.X[col].to_numpy()

# create the subplots with 4 rows and 2 columns
fig,ax = plt.subplots(4,2,figsize=(20,8))  

# call the plotting function with the needed parameters
plot_feature_wise_subplots(X_feature_dict,Y,Y_prime,"Age",2)

# use tight_layout so you can see all the data and it doesnt over lap and finally show the subplots
plt.tight_layout()
plt.show()