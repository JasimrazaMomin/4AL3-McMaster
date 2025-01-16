import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# check if cuda is available else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# define a cnn
class CNN(nn.Module):
    def __init__(self):
        super().__init__()  # call the base class constructor
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)  # first convolutional layer
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=3)  # second convolutional layer
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3)  # third convolutional layer
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)  # max pooling layer
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # first fully connected layer
        self.fc2 = nn.Linear(120, 84)  # second fully connected layer
        self.fc3 = nn.Linear(84, 10)  # output layer
    
    # define the forward pass
    def forward(self, x):
        x = self.conv1(x)  # apply first conv
        x = F.relu(x)  # apply relu 
        x = self.pooling(x)  # apply max pooling
        x = self.conv2(x)  # apply second conv
        x = F.relu(x)  # apply relu 
        x = self.conv3(x)  # apply third conv
        x = F.relu(x)  # apply relu 
        x = self.pooling(x)  # apply max pooling
        x = x.view(-1, 16 * 4 * 4)  # flatten the tensor
        x = self.fc1(x)  # apply first fc
        x = F.relu(x)  # apply relu 
        x = self.fc2(x)  # apply second fc
        x = F.relu(x)  # apply relu 
        x = self.fc3(x)  # apply final fc and output
        return x  # return the output

#define a logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1) # define a linear layer with input_dim features and 1 output
        self.sigmoid = nn.Sigmoid() # define a sigmoid activation function
    
    def forward(self, x):
        output = self.sigmoid(self.linear(x)) # apply the linear layer followed by the sigmoid activation
        return output

def part1():
    # function to load and preprocess data
    def data(batch_size):
        # normalize and transform dataset
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # download and prepare the FashionMNIST training and test datasets
        train_dataset = datasets.FashionMNIST(
            root='./data',
            train=True,
            transform=normalize,
            download=True
        )
        
        test_dataset = datasets.FashionMNIST(
            root='./data',
            train=False,
            transform=normalize,
            download=True
        )

        # split the training data into training and validation sets
        len_train = int(0.8 * len(train_dataset))
        len_val = len(train_dataset) - len_train
        train_data_split, val_data_split = random_split(train_dataset, [len_train, len_val])
        
        # create data loaders for training testing and validation
        train_data = DataLoader(train_data_split, batch_size=batch_size, shuffle=True)
        test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_data = DataLoader(val_data_split, batch_size=batch_size, shuffle=False)
        
        return train_data, val_data, test_data

    # function to train the model
    def train(learning_rate, train_data, val_data, epochs):
        # initialize the CNN model
        model = CNN()
        # define the loss function and optimizer
        cross_entropy = nn.CrossEntropyLoss()
        sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # initialize lists to store losses
        train_loss_list = []
        val_loss_list = []

        for epoch in range(epochs):
            # set the model to training mode
            model.train()
            train_loss = 0
            for image, label in train_data:
                sgd.zero_grad()
                outputs = model(image)
                loss = cross_entropy(outputs, label)
                loss.backward()
                sgd.step()
                train_loss += loss.item()
            
            # evaluate on the validation data
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for image, label in val_data:
                    outputs = model(image)
                    loss = cross_entropy(outputs, label)
                    val_loss += loss.item()

            # calculate average losses
            train_loss = train_loss / len(train_data)
            train_loss_list.append(train_loss)
            val_loss = val_loss / len(val_data)
            val_loss_list.append(val_loss)

            # print loss for each epoch
            print(f"On Epoch {epoch} With Train Loss = {train_loss} And Val Loss = {val_loss}")

        return model, train_loss_list, val_loss_list, epoch

    # function to plot training and validation losses
    def plot(train_loss_list, val_loss_list, epochs):
        plt.figure()
        plt.plot(range(epochs + 1), train_loss_list, label='Training Loss')
        plt.plot(range(epochs + 1), val_loss_list, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    # function to evaluate model accuracy on test data
    def evaluate(model, test_data):
        model.eval()
        right_number = 0
        total_number = 0
        
        with torch.no_grad():
            for image, label in test_data:
                outputs = model(image)
                _, prediction = torch.max(outputs.data, 1)
                total_number = total_number + label.size(0)
                right_number = right_number + (prediction == label).sum().item()
        
        # calculate and print model accuracy
        accuracy = right_number / total_number
        print(f'Model accuracy: {accuracy * 100}')
        return 

    # define hyperparameters
    batch_size = 16
    learning_rate = 0.01
    epochs = 20

    # load the data
    train_data, val_data, test_data = data(batch_size)

    # train the model
    model, train_loss_list, val_loss_list, epoch_number = train(learning_rate, train_data, val_data, epochs)

    # evaluate the model
    evaluate(model, test_data)

    # plot the loss curves
    plot(train_loss_list, val_loss_list, epoch_number)

    plt.savefig(f'{batch_size}_{learning_rate}.png')
    plt.close()
    return

def part2():
    # calculates true positive rate (tpr) and false positive rate (fpr) from confusion matrix
    def equalized_odds(y_true, y_pred):
        tp, fp, fn, tn = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn + 1e-6)
        fpr = fp / (fp + tn + 1e-6)
        return tpr, fpr

    # determines the most disadvantaged race group based on fairness metrics
    def get_biased_race(race_dict):
        disparity = {group: 0 for group in race_dict.keys()}
        for race1 in race_dict.keys():
            for race2 in race_dict.keys():
                if race1 != race2:
                    tpr = abs(race_dict[race1][0] - race_dict[race2][0])
                    fpr = abs(race_dict[race1][1] - race_dict[race2][1])
                    disparity[race1] += (tpr + fpr)
        
        biased_race = max(disparity, key=disparity.get)
        return biased_race
    
    # prints tpr and fpr for a specific race group along with other statistics
    def print_fpr_tpr(Y_true, Y_pred, group, fpr, tpr):
        tn, fp, fn, tp = confusion_matrix(Y_true,Y_pred).ravel()
        print(f"\n\nFor {group}:")
        print(f"Total cases: {len(Y_true)}, Chance of recidivism: {((tp + fp)/len(Y_true)):.2%}, False positive rate: {(fpr):.2%}, True positive rate: {(tpr):.2%}")
    
    # balances the dataset for a specific race group by sampling equally from each class
    def unbiased_data(df, race):
        not_equal_df = df[df['race'] == race]
        
        class_0 = not_equal_df[not_equal_df['recid_score'] == 0]
        class_1 = not_equal_df[not_equal_df['recid_score'] == 1]
        
        if len(class_0) > len(class_1):
            class_0 = class_0.sample(n=len(class_1), random_state=42)
        elif len(class_1) > len(class_0):
            class_1 = class_1.sample(n=len(class_0), random_state=42)

        unbiased_data = pd.concat([class_0, class_1])
        return unbiased_data
    
    # load and preprocess the data
    file_path = "compas-scores.csv"
    df = pd.read_csv(file_path)
    df['recid_score'] = df['score_text'].map({'Low': 0, 'Medium': 0, 'High': 1})
    columns = ['age_cat', 'race', 'c_charge_degree', 'recid_score', 'sex', 'is_recid', 'r_charge_degree', 'decile_score', 'c_charge_desc', 'age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']
    df_selected = df[columns].dropna()
    
    # split into features and target
    Y = df_selected['recid_score']
    X = df_selected.drop(columns=['recid_score'])
    
    # define categorical and numerical features
    cat_features = ['age_cat', 'race', 'c_charge_degree', 'c_charge_desc', 'sex', 'is_recid', 'r_charge_degree']
    num_features = ['age', 'priors_count', 'decile_score', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']
    
    # encode categorical features and scale numerical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    scaler = StandardScaler()
    
    X_cat = encoder.fit_transform(X[cat_features])
    X_num = scaler.fit_transform(X[num_features].values.reshape(-1, len(num_features)))    
    
    # combine categorical and numerical features into a single array
    X_npy = np.hstack([X_cat, X_num])
    X_df = pd.DataFrame(X_npy, index=df_selected.index)
    X_train, X_test, Y_train, Y_test = train_test_split(X_npy, Y, test_size=0.2, random_state=42)
    
    # initialize training parameters and the model
    epochs = 100
    learning_rate = 0.01
    
    model = LogisticRegression(X_train.shape[1])
    bce = nn.BCELoss()
    adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # convert data to tensors for training
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # train the model without bias mitigation
    print("No bias mitigation ------")
    for epoch in range(epochs):
        model.train()
        adam.zero_grad()
        outputs = model(X_train_tensor)
        loss = bce(outputs, Y_train_tensor)
        loss.backward()
        adam.step()
        
        if (epoch) % 10 == 0:
            print(f"Epoch {epoch} with Loss: {loss.item()}")
    
    # evaluate the model on training and testing data
    model.eval()
    with torch.no_grad():
        model_output_train = model(X_train_tensor)
        model_output_test = model(X_test_tensor)
        y_pred_train = (model_output_train >= 0.5).float()
        y_pred_test = (model_output_test >= 0.5).float()

    train_acc = accuracy_score(Y_train, y_pred_train.numpy())
    test_acc = accuracy_score(Y_test, y_pred_test.numpy())
    print(f"Training Accuracy: {train_acc:.2f} And Testing Accuracy: {test_acc:.2f}")      

    # compute fairness metrics per race group
    races = dict()
    groups = df_selected.groupby('race')
    for race, data in groups:
        indices = data.index
        
        Y_true = Y.loc[indices] 
        X_grouped = X_df.loc[indices]
        
        model_input = torch.tensor(X_grouped.values, dtype=torch.float32)
        with torch.no_grad():
            model_output = model(model_input)
            Y_pred = (model_output >= 0.5).int().flatten().numpy()
        
        tpr, fpr = equalized_odds(Y_true, Y_pred)
        races[race] = (tpr,fpr)
        print_fpr_tpr(Y_true, Y_pred, race, fpr, tpr)
    
    biased = get_biased_race(races)
    print(biased)
    
    # balance the data for the most disadvantaged group
    bal_data = unbiased_data(df_selected, biased)
    df_without_race = df_selected[df_selected['race'] != biased]
    df_less_biased = pd.concat([df_without_race, bal_data]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # re-encode and rescale balanced data
    Y_ = df_less_biased['recid_score']
    X_ = df_less_biased.drop(columns=['recid_score'])
    X_cat_bal = encoder.transform(X_[cat_features])
    X_num_bal = scaler.transform(X_[num_features])
    X_npy_bal = np.hstack([X_cat_bal, X_num_bal])
    
    X_df_bal = pd.DataFrame(X_npy_bal, index=df_less_biased.index)
    X_train, X_test, Y_train, Y_test = train_test_split(X_npy_bal, Y_, test_size=0.2, random_state=42)
    X_train_bal_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_bal_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_train_bal_tensor = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)
    
    # retrain the model with bias mitigation
    model = LogisticRegression(X_train.shape[1])
    bce = nn.BCELoss()
    adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("With bias mitigation ------")
    for epoch in range(epochs):
        model.train()
        adam.zero_grad()
        outputs = model(X_train_bal_tensor)
        loss = bce(outputs, Y_train_bal_tensor)
        loss.backward()
        adam.step()

        if (epoch) % 10 == 0:
            print(f"Epoch {epoch} with Loss: {loss.item()}")
           
    # evaluate the less biased model 
    model.eval()
    with torch.no_grad():
        model_output_train_bal = model(X_train_bal_tensor)
        model_output_test_bal = model(X_test_bal_tensor)
        y_pred_train_bal = (model_output_train_bal >= 0.5).float()
        y_pred_test_bal = (model_output_test_bal >= 0.5).float()
        
        train_accuracy_bal = accuracy_score(Y_train, y_pred_train_bal.numpy())
        test_accuracy_bal = accuracy_score(Y_test, y_pred_test_bal.numpy())
        print(f"Less Biased Training Accuracy: {train_accuracy_bal:.2f}")
        print(f"Less Biased Testing Accuracy: {test_accuracy_bal:.2f}\n")
        
        model_output_bal = model(torch.tensor(X_npy_bal, dtype=torch.float32))
        y_pred_bal = (model_output_bal >= 0.5).int().flatten().numpy()

    # recompute fairness metrics after bias mitigation
    races = dict()
    groups = df_less_biased.groupby('race')
    for race, data in groups:
        group_indices = data.index
        Y_true = Y_.iloc[group_indices]
        Y_pred = y_pred_bal[group_indices]
        
        tpr, fpr = equalized_odds(Y_true, Y_pred)
        races[race] = (tpr,fpr)
        print_fpr_tpr(Y_true, Y_pred, race, fpr, tpr)
    
    print(get_biased_race(races))
    return

print("Part 1 starts here\n\n\n")
part1()
print("\n\n\nPart 2 starts here\n\n\n")
part2()