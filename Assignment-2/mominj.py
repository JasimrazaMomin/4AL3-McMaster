import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sn

class SVM:
    # initialize the svm class using the x feature matrix, y output vector, c hyperparameter, and set the normalized x and ys to none and set my own svm function
    def __init__(self, X, Y, C=1):
        self.X = X
        self.Y = Y
        self.C = C 
        self.X_prime = None
        self.Y_prime = None
        self.svm = SVC(C=self.C, kernel="rbf", gamma='scale') # using this since it performed best compared to using other kernels and gamma parameters

    # using standard scaler to normalize my x feature matrix and converting my y output vector to a numpy array
    def preprocess(self):
        scaler = StandardScaler()
        self.X_prime = scaler.fit_transform(self.X) # does the X - mean divided by std for us and outputs a numpy array that we can use
        self.Y_prime = self.Y.to_numpy()
        return None
    
    # using a list of fs numbers, we concatenate the features to our data and set our own feature matrix equal to it 
    def feature_creation(self, chosen_features):
        selected_data = pd.DataFrame()
        for feature in chosen_features:
            if feature == 1:
                selected_data = pd.concat([selected_data, self.X.iloc[:,:18]],axis=1) # fs 1
            elif feature == 2:
                selected_data = pd.concat([selected_data, self.X.iloc[:,18:90]],axis=1) # fs 2
            elif feature == 3:
                selected_data = pd.concat([selected_data, self.X.iloc[:,90]],axis=1) # fs 3
            elif feature == 4:
                selected_data = pd.concat([selected_data, self.X.iloc[:,91:]],axis=1) # fs 4
        self.X = selected_data 
        return None

    # do k fold cross validation and return the average accuracy, a list of the tss scores we got, and a list of the confusion matrices
    def cross_validation(self, k, shuffle, random):
        folds = KFold(n_splits=k, shuffle=shuffle, random_state=random) # get our k fold indices from sklearns KFold
        accuracy_list = [] # make our lists to track our accuracy, tss, and confusion matrices
        tss_list = []
        confusion_matrix_list = []
        
        for train_index, test_index in folds.split(self.X_prime): 
            X_train, X_test = self.X_prime[train_index], self.X_prime[test_index] # split our X data into train and test
            Y_train, Y_test = self.Y_prime[train_index], self.Y_prime[test_index] # split our Y data into train and test
            
            self.training(X_train, Y_train) # call training function
            Y_pred = self.svm.predict(X_test) # do the prediction on the test set
            
            accuracy_list += [accuracy_score(Y_test, Y_pred)] # get the accuracy score and add it
            tss_list += [self.tss(Y_test, Y_pred)] # get the tss score from the tss function 
            confusion_matrix_list += [confusion_matrix(Y_test, Y_pred)] # get the confusion matrix and add it

        average_acc = sum(accuracy_list)/len(accuracy_list) # average out the accuracy and return all three in a list
        return [average_acc, tss_list, confusion_matrix_list]
    
    # train the svm on the provided X and Y data
    def training(self,X,Y):
        self.svm.fit(X,Y) # call the svms built in fit function
        return None
    
    # perform the tss calculation based on the inputed true and predicted values and return it
    def tss(self, Y_true, Y_pred):
        cm = confusion_matrix(Y_true, Y_pred) # get the confusion matrix
        tn = cm[0,0] # get the true negatives from the confusion matrix
        fp = cm[0,1] # get the false positives from the confusion matrix
        fn = cm[1,0] # get the false negatives from the confusion matrix
        tp = cm[1,1] # get the true positives from the confusion matrix
        
        # since the tss equation is in two parts, we calculate it in two parts 
        tp_half = 0 # set it to 0 incase the denominator is non-positive
        if (tp + fn) > 0: # check if denominator is non-positive
            tp_half = tp / (tp + fn) # if the denominator is positive, we perform the first half of the tss calculation
            
        fp_half = 0 # set it to 0 incase the denominator is non-positive
        if (tn + fp) > 0: # check if denominator is non-positive
            fp_half = fp / (tn + fp) # if the denominator is positive, we perform the second half of the tss calculation

        return tp_half - fp_half # return the tss by subtracting the second half from the first half (as seen in the assignment file)

# gets the x feature matrix and y output vector based on the directory, also dictates whether follow data order or not
def get_all_feature_output(directory, data_order):
    # load the data from the numpy array files based on the passed in directory
    pos_features_main_timechange = np.load(f"data-{directory}/pos_features_main_timechange.npy", allow_pickle=True)
    neg_features_main_timechange = np.load(f"data-{directory}/neg_features_main_timechange.npy", allow_pickle=True)
    pos_features_historical = np.load(f"data-{directory}/pos_features_historical.npy", allow_pickle=True)
    neg_features_historical = np.load(f"data-{directory}/neg_features_historical.npy", allow_pickle=True)
    pos_features_maxmin = np.load(f"data-{directory}/pos_features_maxmin.npy", allow_pickle=True)
    neg_features_maxmin = np.load(f"data-{directory}/neg_features_maxmin.npy", allow_pickle=True)
    pos_class = np.load(f"data-{directory}/pos_class.npy", allow_pickle=True)
    neg_class = np.load(f"data-{directory}/neg_class.npy", allow_pickle=True)

    # column stack all the features and at the start add in the classification vector, so 1 for positive, -1 for negative
    fs_pos = np.column_stack((np.ones(pos_class.shape[0]),pos_features_main_timechange, pos_features_historical, pos_features_maxmin))
    fs_neg = np.column_stack((np.full(neg_class.shape[0],-1),neg_features_main_timechange, neg_features_historical, neg_features_maxmin))
    
    # combine the positive and negative feature combinations 
    df = pd.concat([pd.DataFrame(fs_pos), pd.DataFrame(fs_neg)], axis=0).reset_index(drop=True) 
    
    # if we need to follow data order, we load the file and reorder based on the order in there
    if data_order:
        data_order_npy = np.load(f"data-{directory}/data_order.npy")
        df = df.iloc[data_order_npy]

    X = df.iloc[:,1:] # use iloc to split the df into the feature matrix 
    Y = df.iloc[:,0] # use iloc get df the output vector (first column)
    return X, Y # return the x feature matrix and y column vector

# gets the powerset based on a list s 
def power_set(s):
    powerset = [[]] # empty set in the powerset
    for x in s:
        # for each element add it to all the subsets that are there
        new_subsets = [subset + [x] for subset in powerset]
        powerset += new_subsets # add the new subsets to the powerset
    return powerset # return the powerset

# plot a line graph for the tss scores for the two datasets across folds
def visualize_tss_per_dataset(tss_list, datasets):
    plt.figure(figsize=(20, 8))
    
    # plot the tss score for each dataset
    for i, dataset in enumerate(datasets):
        plt.plot(tss_list[i], marker='x', label=f"Dataset {dataset}")
        
    # add the title and labels
    plt.title("TSS for Both Datasets")
    plt.xlabel("Fold")
    plt.ylabel("TSS")
    plt.ylim(0,1) # set the limits so that you can see the tss between 0 and 1
    
    # add a legend to see what feature combination is what and set the location and font size and show
    plt.legend(fontsize=12,loc=3)
    plt.show()
    return None

# plot a line graph for the tss scores for different feature combinations across folds
def visualize_tss_feature_combos(tss_list, feature_combos):
    plt.figure(figsize=(20, 8))
    
    # plot the tss score for every feature combination
    for i, combination in enumerate(feature_combos):
        plt.plot(tss_list[i], marker='x', label=f"Features {combination}")
    
    # add the title and labels
    plt.title("TSS for Feature Combinations")
    plt.xlabel("Fold")
    plt.ylabel("TSS")
    plt.ylim(-0.05,1) # set limits in a way that you can see the tss of 0 (if set to 0 then it blends with the x axis)
    
    # add a legend to see what feature combination is what and set the location and font size and show
    plt.legend(fontsize=10,loc=3)
    plt.show()
    return None

# function to plot the confusion matrix 
def plot_confusion_matrix(all_matrices, title):
    summed_matrices = np.sum(all_matrices,axis=0) # sum all the confusion matrices across the folds
    
    plt.figure(figsize=(9, 6)) 
    sn.heatmap(summed_matrices, annot=True, fmt="d", cmap="Blues", cbar=False) # plot the heatmap using seaborn
    
    # based on the title, set the title for the graph
    if title == "2010-15":
        plt.title("Confusion Matrix Using Dataset {}".format(title))
    elif title == "2020-24":
        plt.title("Confusion Matrix Using Dataset {}".format(title))
    elif title == "Data Order":
        plt.title("Confusion Matrix Using {}".format(title))
    else:
        plt.title("Confusion Matrix of FS-{}".format(title))
    
    # set labels and show
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return None

# function to perform an experiement using the best feature combination and using the data order file
def use_data_order(best_feature_combination):
    X, Y = get_all_feature_output("2010-15", True) # get the x feature matrix anad y output vector for the directory, also make sure it uses data order by passing in True
    svm = SVM(X, Y, C=1) # initialize an instance of the svm using the x feature matrix, y output vector, and a C value of 1
    svm.feature_creation(best_feature_combination) # turn x into the feature combination that we passed in
    svm.preprocess() # normalize the data
    
    # get the result of the cross validation and split it up into its three parts
    cv_result = svm.cross_validation(10, False, None)
    average_accuracy = cv_result[0]
    tss = cv_result[1]
    confusion_matrix_list = cv_result[2]
    
    # get the average tss
    average_tss = sum(tss)/len(tss)
    
    # print the average accuracy and average tss 
    print("Using Data Order:")
    print(f"Average Accuracy: {average_accuracy}")
    print(f"Average TSS: {average_tss}")
    
    # plot the confusion matrix with the title Data Order
    plot_confusion_matrix(confusion_matrix_list, "Data Order")
    return None

# perform the feature experiment
def feature_experiment():
    X, Y = get_all_feature_output("2010-15",False) # get the x feature matrix and y output vector
    feature_combinations = [list(x) for x in power_set([1,2,3,4])[1:]] # get all feature combinations
    tss_list = [] # use a list to track each feature combinations tss 
    best_tss = 0 # use this to find the highest tss
    best_feature_combination = [] # use this to track the best feature combination
    for features in feature_combinations:
        svm = SVM(X, Y, C=1) # initialize an instance of the svm using the x feature matrix, y output vector, and a C value of 1
        svm.feature_creation(features) # turn x into the feature combination that we chose
        svm.preprocess() # normalize the data 
        
        # get the result of the cross validation and split it up into its three parts
        cv_result = svm.cross_validation(10,True,42) 
        average_accuracy = cv_result[0]
        tss = cv_result[1]
        confusion_matrix_list = cv_result[2]
        
        # add the list of tss, get the average tss and see if it is the highest tss
        tss_list.append(tss)
        avg_tss = sum(tss)/len(tss)
        if avg_tss > best_tss:
            best_feature_combination = features
            best_tss = avg_tss
        
        # print out the feature combination, average accuracy, and average tss
        print(f"Feature Combination With FS-{features}:")
        print(f"Average Accuracy: {average_accuracy}")
        print(f"Average TSS: {avg_tss}\n")
        
        # plot the confusion matrix for this feature combination
        plot_confusion_matrix(confusion_matrix_list, features)
    visualize_tss_feature_combos(tss_list, feature_combinations) # plot a line graph with all tss values across all feature combinations and folds
    print(f"Best Feature Combinations: FS-{best_feature_combination}\n") # print out the best feature combination that was found (highest tss)
    return best_feature_combination # return the best feature combination

# perform the data experiment
def data_experiment(best_feature_combination):
    directories = ['2010-15', '2020-24'] # make a list of the two directories to iterate over
    tss_list = [] # use a list to track each directorys tss 
    for directory in directories:
        X, Y = get_all_feature_output(directory,False) # get the x feature matrix anad y output vector for the directory
        svm = SVM(X, Y, C=1) # initialize an instance of the svm using the x feature matrix, y output vector, and a C value of 1
        svm.feature_creation(best_feature_combination) # turn x into the feature combination that we passed in
        svm.preprocess() # normalize the data       
         
        # get the result of the cross validation and split it up into its three parts
        cv_result = svm.cross_validation(10,True,42)
        average_accuracy = cv_result[0]
        tss = cv_result[1]
        confusion_matrix_list = cv_result[2]
        
        # add the list of tss and get the average tss
        tss_list.append(tss)
        average_tss = sum(tss)/len(tss)
        
        # print out the directory used, feature combination used, average accuracy, and the average tss
        print(f"Dataset data-{directory}:")
        print(f"Feature Combination with FS-{best_feature_combination}")
        print(f"Average Accuracy: {average_accuracy}")
        print(f"Average TSS: {average_tss}\n")
        
        # plot the confusion matrix for the directory
        plot_confusion_matrix(confusion_matrix_list, directory)
    visualize_tss_per_dataset(tss_list, directories) # plot a line graph with the two directory tss values across each fold
    return None

# first, get the best feature combination from the feature experiment run on directory data-2010-15
# since we go through each and every feature combination, we select the best one based off of the highest tss value/score
best_feature_combination = feature_experiment()    

# once we get the best feature combination, we use it to perform the data experiment where we see how the model performs on both datasets
data_experiment(best_feature_combination)

# finally, we test out the best feature combination on the 2010 dataset using the data order file to see what happens and how it performs
use_data_order(best_feature_combination)