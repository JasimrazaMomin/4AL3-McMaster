
# Author: Jasimraza Momin
# Created: November 8, 2024
# License: MIT License
# Purpose: This python file includes boilerplate code for Assignment 3

# Usage: python mominj_part1-2-3.py

# Dependencies: None
# Python Version: 3.6+

# Modification History:
# - Version 1 - added boilerplate code

# References:
# - https://www.python.org/dev/peps/pep-0008/
# - Python Documentation

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

np.random.seed(42) #to keep the same seed on each run

#I dont know what is actually wanted in terms of results, so I aimed to optimize my accuracy since that is usually the best indicator of model performance
#I know that we want to show trends in the loss and to do that I changed my C value, but changing it too much for part 3 resulting in almost unusable accuracy <10%
#As such I used what C and LR values worked for part 1 and 2 there and used different oens for part 3
#From part 1 and part 2 you can see the trend in the loss and a decent performance, for part 3 the convergence gets us a good accuracy but the graph doesnt have a well defined loss trend
#This is just a little description before you take a look at my code and run it so you know whats going on, I asked a TA and there wasnt much clarification as to what
#I needed to optimize so please keep that in mind when marking

class svm_():
    def __init__(self,learning_rate,epoch,C_value,X,Y,X_,Y_):

        #initialize the variables to store training and testing data
        self.input = X 
        self.target = Y
        self.test_input = X_ 
        self.test_output = Y_
        #initialize the variables for the learning rate, epochs, and c values
        self.learning_rate =learning_rate
        self.epoch = epoch
        self.C = C_value
        #initialize 2 dicts to keep track of losses
        self.train_loss_dict = dict()
        self.val_loss_dict = dict()
        #initialize weights randomly and initialize variables used in early stopping
        self.weights = np.random.randn(X.shape[1])
        self.stopped = None
        self.stop_weights = None
        
    def pre_process(self,): #kept same as given
        #using StandardScaler to normalize the input
        scalar = StandardScaler().fit(self.input)
        X_ = scalar.transform(self.input,copy=True)

        Y_ = self.target 
        
        return X_,Y_ 
    
    # the function return gradient for 1 instance -
    # stochastic gradient decent
    def compute_gradient(self,X,Y): #kept same as given
        # organize the array as vector
        X_ = np.array([X])

        # hinge loss
        hinge_distance = 1 - (Y* np.dot(X_,self.weights))

        total_distance = np.zeros(len(self.weights))

        # hinge loss is not defined at 0
        # is distance equal to 0
        if max(0, hinge_distance[0]) == 0:
            total_distance += self.weights
        else:
            total_distance += self.weights - (self.C * Y * X_[0])

        return total_distance

    def compute_loss(self,X,Y):
        # calculate hinge loss
        # hinge loss implementation- start
        # Part 1
        
        loss = 0 #initialize loss
        reg = 0.5 * np.linalg.norm(self.weights) ** 2 #calculate regularized weights
        for i in range(X.shape[0]): #iterate over each row index
            loss += max(0, 1 - Y[i] * np.dot(X[i], self.weights)) #calculate the prediction and get the max in terms of the loss and 0
        # hinge loss implementatin - end
        
        return self.C * loss + reg #return the hinge loss
    
    def stochastic_gradient_descent(self,X,Y,early_stop=True): #added in an early stop flag
        last_loss = float('inf') #keep last loss as infinity to start with
        delta_loss = 1e-3 #delta loss acts as our threshold difference
        not_stopped = True #for early stopping
        temp_stop = 0 #for early stopping
        # execute the stochastic gradient des   cent function for defined epochs
        for epoch in range(self.epoch): #iterate over each epoch

            # shuffle to prevent repeating update cycles
            features, output = shuffle(X, Y) #shuffle x and y

            for i, feature in enumerate(features): #go over each feature and calculate the gradient
                gradient = self.compute_gradient(feature, output[i])
                self.weights = self.weights - (self.learning_rate * gradient)

            #check for convergence -start
            loss = self.compute_loss(features, output) #compute training loss
            val_loss = self.compute_loss(self.test_input,self.test_output) #compute validation loss
            # Part 1

            if epoch%(self.epoch//10)==0: #if the epoch is a 1/10th of the total epochs, we plot it 
                print("Epoch is: {} and Train Loss is : {}".format(epoch, loss)) #printing out the training loss
                print("Epoch is: {} and Validation Loss is : {}\n".format(epoch, val_loss)) #printing out the validation loss
                self.train_loss_dict[epoch] = loss #add loss and epoch to the dictionary
                self.val_loss_dict[epoch] = val_loss #add loss and epoch to the dictionary
                if not_stopped: #check if we havent stopped
                    temp_stop = epoch #set our temp stop (for graphing)
                
            if abs(last_loss - loss) <= delta_loss and self.stopped is None and early_stop: #check if we are supposed to stop and if the threshold has been hit and if we have no stopped already (converged)
                print("The minimum number of iterations taken are:",epoch) #print out the number of iterations taken
                self.stopped = epoch #store when we stopped
                not_stopped = False #make sure we indicate that we have stopped 
                self.stop_weights = self.weights #set our stop weights so we can use them to predict later
            
            last_loss = loss #set last loss to this loss for next epoch
            
            #check for convergence - end
            #added all part 3 logic to part 3 instead of sampling here
        if self.stopped is None: #if we had not stopped (converged) then we set our stopped to be our last graphing point (for the stopping line)
            self.stopped = temp_stop #set to last graphed loss at final epoch

        print("Training ended...")
        print("weights are: {}".format(self.weights)) #print out our weights

    def mini_batch_gradient_descent(self,X,Y,batch_size):

        # mini batch gradient decent implementation - start
        for epoch in range(self.epoch): #iterate over each epoch

            features, output = shuffle(X, Y) #shuffle our x and y

            for i in range(0, len(features), batch_size): #go for i from 0 to length of features and skip by batch_size to get batch indexes
                j = min(i + batch_size, len(features)) #to make sure we dont overstep the end (out of index errors)
                batch_features = features[i:j] #get our features
                batch_output = output[i:j] #get our outputs

                batch_gradient = np.zeros(len(self.weights)) #initialize this to track our batch gradient
                for k in range(len(batch_features)):
                    batch_gradient += self.compute_gradient(batch_features[k], batch_output[k]) #compute the gradient for each instance in the batch

                batch_gradient /= len(batch_features) #average out the gradient
                self.weights = self.weights - (self.learning_rate * batch_gradient) #update the weights with the batch gradient

            if epoch%(self.epoch//10)==0: #if the epoch is a 1/10th of the total epochs, we plot it 
                loss = self.compute_loss(features, output) #calculate the training loss
                val_loss = self.compute_loss(self.test_input,self.test_output) #calculate the validation loss
                print("Epoch is: {} and Train Loss is : {}".format(epoch, loss)) #printing out the training loss
                print("Epoch is: {} and Validation Loss is : {}\n".format(epoch, val_loss)) #printing out the validation loss
                self.train_loss_dict[epoch] = loss #store the loss at epoch
                self.val_loss_dict[epoch] = val_loss #store hte loss at epoch
                
        # Part 2

        # mini batch gradient decent implementation - end

        print("Training ended...")
        print("weights are: {}".format(self.weights)) #print out the weights
    
    def sampling_strategy(self,unlabelled_x,unlabelled_y):
        #implementation of sampling strategy - start
        #Part 3
        #zip together the unlabelled stuff and iterate over each to get each instances loss
        loss_list = [float(self.compute_loss(np.array([input]), np.array([test]))) for input, test in zip(unlabelled_x, unlabelled_y)]
        loss_index = np.argmax(loss_list) #using argmax since it gave faster convergence (prof said to try it out and it work lol)
        
        #implementation of sampling strategy - start
        return loss_index #return the index for the sampling

    def predict(self,X_test,Y_test):

        #compute predictions on test set
        predicted_values = [np.sign(np.dot(X_test[i], self.weights)) for i in range(X_test.shape[0])]
        
        #compute accuracy
        accuracy= accuracy_score(Y_test, predicted_values)
        print("Accuracy: {}".format(accuracy))
        #compute precision - start
        # Part 2
        precision = precision_score(Y_test,predicted_values) #compute precision
        print("Precision: {}".format(precision))
        
        #compute precision - end

        #compute recall - start
        # Part 2
        recall = recall_score(Y_test,predicted_values) #compute recall
        print("Recall: {}".format(recall))
        
        #compute recall - end
        return accuracy, precision, recall #return accuracy precision and recall

def plot_loss_2(train_loss_dict,val_loss_dict,train_loss_dict2,val_loss_dict2): #plotting function for part 2
    #split first dict into X (epochs as keys) and Y (loss as values)
    X = list(train_loss_dict.keys())
    Y = list(train_loss_dict.values())
    #split first dict into X_ (epochs as keys) and Y_ (loss as values)
    X_ = list(val_loss_dict.keys())
    Y_ = list(val_loss_dict.values())
    
    #split first dict into X (epochs as keys) and Y (loss as values)
    X2 = list(train_loss_dict2.keys())
    Y2 = list(train_loss_dict2.values())
    #split first dict into X_ (epochs as keys) and Y_ (loss as values)
    X_2 = list(val_loss_dict2.keys())
    Y_2 = list(val_loss_dict2.values())
    
    #plot
    plt.plot(X,Y,label="Train(Mini)",marker='x')
    plt.plot(X_,Y_,label="Validation(Mini)",marker='o')
    plt.plot(X2,Y2,label="Train(SGD)",marker='x')
    plt.plot(X_2,Y_2,label="Validation(SGD)",marker='o')
    #plot early stopping line and easier way to set title as 
    plt.title("Losses for Mini Batch Vs SGD")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.legend()
    plt.show()
    return

def plot_loss_1(train_loss_dict,val_loss_dict,early_stopping): #plotting function for part 1
    #split first dict into X (epochs as keys) and Y (loss as values)
    X = list(train_loss_dict.keys())
    Y = list(train_loss_dict.values())
    #split first dict into X_ (epochs as keys) and Y_ (loss as values)
    X_ = list(val_loss_dict.keys())
    Y_ = list(val_loss_dict.values())
    
    #plot
    plt.plot(X,Y,label="Train",color='r',marker='x')
    plt.plot(X_,Y_,label="Validation",color='g',marker='o')
    plt.axvline(x=early_stopping, color='b', label='Stopped Here')
    plt.title("Losses for Early Stopping") 
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.legend()
    plt.show()
    return

def plot_loss_3 (big_dict1,big_dict2,samples): #plotting function for part 3
    plt.figure(figsize=(20,8)) #set big figure size to make sure legend fits
    for key in big_dict1.keys(): #go over each key which represents the sample at which the interal dicts were taken
        X = list(big_dict1[key].keys()) #training dict epochs
        Y = list(big_dict1[key].values()) #training dict loss
        X_2 = list(big_dict2[key].keys()) #val dict epoch
        Y_2 = list(big_dict2[key].values()) #val dict loss
        plt.plot(X,Y,label=f'Train {key}',marker='x') #legend for training at sample
        plt.plot(X_2,Y_2,label=f'Validation {key}',marker='o') #legend for validation at sample
    #plot
    plt.title(f"Losses for Used Samples = {samples}") 
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=9,loc = 1)
    plt.show()
    return

def part_1(X_train,y_train,X_test,Y_test,c,lr): #pass in training and testing sets along with c and lr values
    #model parameters - try different ones
    C = c #set c value
    learning_rate = lr #set learning rate
    epoch = 100 #set epochs
  
    #intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train,X_=X_test,Y_=Y_test) 

    #pre preocess data
    X,Y= my_svm.pre_process()
    # select samples for training
    # train model
    my_svm.stochastic_gradient_descent(X,Y)
    plot_loss_1(my_svm.train_loss_dict,my_svm.val_loss_dict,early_stopping=my_svm.stopped) #plot the loss
    my_svm.weights = my_svm.stop_weights #set my weights to the stopped weights (to replicate the prediction that should come from early stopping)
    #to add more context the prof wants us to plot beyond the stopping point to see the change in loss (plateaus or decreases etc), so i stored the weights
    #when we stopped and then now set the weights to those stopped weights to see how early stopping affects our accuracy
    return my_svm #return the model

def part_2(X_train,y_train,X_test,Y_test,c,lr,part1_svm): #pass in the training and testing sets along with the c and lr values and part 1 svm
    #model parameters - try different ones
    epoch = 100 #set epochs
    C = c #set c value
    learning_rate = lr #set learning rate
  
    #intantiate the support vector machine class above    
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train,X_=X_test,Y_=Y_test)
    
    #pre preocess data
    X,Y= my_svm.pre_process()
    # select samples for training

    # train model
    my_svm.mini_batch_gradient_descent(X,Y,10)
    plot_loss_2(my_svm.train_loss_dict,my_svm.val_loss_dict,part1_svm.train_loss_dict,part1_svm.val_loss_dict) #plotting function for losses mini vs sgd
    return my_svm #return the model

def part_3(X_,Y_,c,lr): #pass in the training set and the c and lr values
    #model parameters - try different ones
    C = c #set c value
    learning_rate = lr #set learning rate
    epoch = 100 #set epochs
    max_samples = len(X_) #set max samples
    delta_loss = 1e-2 #set delta loss for convergence
    last_loss = float('inf') #set last lost as infinity for convergence check
    used_samples = 10 #set number of used samples
    initial = 10 #set number of initial samples to use
  
    #get random indexes using numpy and get the corresponding x and y instances
    indexes = np.random.choice(len(X_), initial, replace=False) 
    X_initial = X_[indexes]
    Y_initial = Y_[indexes]
    
    #delete those instances from the unlabelled data
    unlabelled_x = np.delete(X_, indexes, axis=0)
    unlabelled_y = np.delete(Y_, indexes, axis=0)
    
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_initial,Y=Y_initial,X_=None,Y_=unlabelled_y) #initialize our svm
    X,Y = my_svm.pre_process() #preprocessed x and y
    #set dicts to be as we will use to store svm dicts later
    train_loss_dict = dict()
    val_loss_dict = dict()
    target_acc = 0.97 #set target accuracy for satisfactory performance
    while used_samples < max_samples and len(unlabelled_x) > 0: #iterate while we havent run out of labelled data and while we are less than our max samples
        
        scaler = StandardScaler().fit(unlabelled_x) #fit scaler for normalization
        normed_x = scaler.transform(unlabelled_x,copy=True) #get normalized data
        
        my_svm.test_input = normed_x #set normalized testing data in svm
        my_svm.stochastic_gradient_descent(X,Y,early_stop=False) #perform sgd without early stopping
        #get dicts from svm and store based on sample number that was used
        train_loss_dict[used_samples] = my_svm.train_loss_dict 
        val_loss_dict[used_samples] = my_svm.val_loss_dict
        
        train_loss = my_svm.compute_loss(X,Y) #get loss for this sample iteration
        
        #check for convergence like in early stopping to see if our sample size is sufficient
        if abs(last_loss - train_loss) <= delta_loss and my_svm.predict(normed_x,unlabelled_y)[0] >= target_acc: 
            #since we need want a good perforamance, we check for both loss convergence and hitting a good target accuracy
            #the accuracy might not be well reflected in the outside test set but it provides a way to know that further testing is not needed
            print(f"Converges with {used_samples} used.") #print out when we converge and break out
            break
        last_loss = train_loss #set last loss as current loss for next sample
        
        index = my_svm.sampling_strategy(normed_x,unlabelled_y) #get index from sampling strategy
        my_svm.target = np.append(my_svm.target,unlabelled_y[index]) #add to svms target data the new instance
        my_svm.input = np.vstack((my_svm.input,[unlabelled_x[index]])) #add to svms input data the new instance
        #remove newly selected data
        unlabelled_x = np.delete(unlabelled_x, index, axis=0)
        unlabelled_y = np.delete(unlabelled_y, index, axis=0)
        #reset the dictionary to let us track the new losses for the new sample
        my_svm.train_loss_dict = dict()
        my_svm.val_loss_dict = dict()
        X,Y = my_svm.pre_process() #preprocess and get x and y for next iteration of sampling
        
        used_samples += 1 #increment used samples
        print(used_samples) #print it out
    #intantiate the support vector machine class above  
    
    #pre preocess data
    # select samples for training

    # train model
    plot_loss_3(train_loss_dict,val_loss_dict,used_samples) #call the plotting function for the samples
    return my_svm, used_samples #return the model and the used samples

#Load datapoints in a pandas dataframe
print("Loading dataset...")
data = pd.read_csv('data1.csv')

# drop first and last column 
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

#segregate inputs and targets

#inputs
X = data.iloc[:, 1:]

#add column for bias
X.insert(loc=len(X.columns),column="bias", value=1)
X_features = X.to_numpy()

#converting categorical variables to integers 
# - this is same as using one hot encoding from sklearn
#benign = -1, melignant = 1
category_dict = {'B': -1.0,'M': 1.0}
#transpose to column vector
Y = np.array([(data.loc[:, 'diagnosis']).to_numpy()]).T
Y_target = np.vectorize(category_dict.get)(Y)

# split data into train and test set using sklearn feature set
print("splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=0.2, random_state=42)

scalar = StandardScaler().fit(X_test) #fit scaler for testing
X_Test_Norm = scalar.transform(X_test,copy=True) #normalize the test data

c_param = 0.01 #c parameters for part 1 and 2 and 3
lr = 0.001 #learning rate for part 1 and 2
lr3 = 0.05 #learning rate for part 3 (higher accuracy for me)

my_svm1 = part_1(X_train,y_train,X_Test_Norm,y_test,c_param,lr) #call part 1 and get the model from part 1

my_svm2 = part_2(X_train,y_train,X_Test_Norm,y_test,c_param,lr,my_svm1) #call part 2 and get the model from part 2

my_svm3,samples = part_3(X_train,y_train,c_param,lr3) #call part 3 and get the model and samples from part 3


# testing the model
#printing out the accuracy precision and recall of the model on the test set
print("\nTesting model 1 accuracy (Test)...")
my_svm1.predict(X_Test_Norm,y_test)
#printing out the epoch early stopping occured
print(f"Stopped at Epoch {my_svm1.stopped}")

#printing out the accuracy precision and recall of the model on the test set
print("\nTesting model 2 accuracy (Test)...")
my_svm2.predict(X_Test_Norm,y_test)

#printing out the accuracy precision and recall of the model on the test set
print("\nTesting model 3 accuracy (Test)...")
my_svm3.predict(X_Test_Norm,y_test)
#printing out the number of samples used
print(f"\nSamples used = {samples}")