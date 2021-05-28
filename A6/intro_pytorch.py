'''
CS540 - Spring 2021
Ayuj Prasad
Homework 6
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from torch import keras

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 50)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 50)
    if training == True:
        return train_loader
    else:
        return test_loader


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784,128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
        )
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    tempval = len(train_loader)
        
    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            
            opt.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            
            running_loss += loss.item()
        percent = (correct/total) * 100
        lossVal = running_loss/tempval
        print("Train Epoch: %d 	 Accuracy: %d/%d(%.2f%%)	Loss: %.3f" % (epoch, correct, total, percent, lossVal))
            
    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss
    percent = correct/total * 100
    lossVal = running_loss/total
    if show_loss == True:
        print("Average loss: %.4f" % (lossVal))
        print("Accuracy: %.2f%%" % (percent))
    else:
        print("Accuracy: %.2f%%" % (percent))
            
            
    

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """

    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    logits = model(test_images)
    prob = F.softmax(logits, dim = 1)
    probVals = (torch.topk(prob[index], 3)[0])
    probVals = probVals.detach().numpy()
    top = torch.topk(prob[index], 3)
    top = top.indices.detach().numpy()
    
    print("%s: %s%%" % (class_names[top[0]], str(round(probVals[0] * 100, 2))))
    print("%s: %s%%" % (class_names[top[1]], str(round(probVals[1] * 100, 2))))
    print("%s: %s%%" % (class_names[top[2]], str(round(probVals[2] * 100, 2))))

    


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()