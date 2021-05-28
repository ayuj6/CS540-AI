'''
Ayuj Prasad
CS 540 - Spring 2021
Assignment 3
'''

from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)  
    x = x - np.mean(x, axis = 0)

    return x

def get_covariance(dataset):
    total = len(dataset)
    covariance = (1 / (total - 1)) * np.dot(np.transpose(dataset), dataset)

    return covariance

def get_eig(S, m):
    total = len(S)
    w, v = eigh(S, eigvals = (total - m, total - 1))
    v = v[:, ::-1]
    w = np.diag(w[::-1])

    return w, v

def get_eig_perc(S, perc):
    maxArr = []
    vects = []
    eigVal, eigVector = eigh(S)
    sum_Val = 0
    sum_Val = sum(eigVal)
    n = len(eigVal)
    for i in range(n):
        if(eigVal[i] / sum_Val) > perc:
            maxArr.append(eigVal[i])
            vects.append(eigVector[:,i])
    maxArr = maxArr[::-1]
    vects = np.transpose(vects)
    vects = np.flip(vects, axis = 1)
    
    return np.diag(maxArr), vects

def project_image(img, U):
    return np.dot(U, np.dot(np.transpose(U), img))

def display_image(orig, proj):
    origr_32 = np.reshape(orig, (32, 32))
    projr_32 = np.reshape(proj, (32, 32))
    fig, (ax1, ax2) = plt.subplots(1, 2)

    bar1 = ax1.imshow(origr_32.T, aspect='equal')
    ax1.set_title("Original")

    bar2 = ax2.imshow(projr_32.T, aspect='equal')
    ax2.set_title("Projection")
    
    fig.subplots_adjust(wspace = 0.45)
    
    fig.colorbar(bar1, ax=ax1, fraction=.045, pad=.075)
    fig.colorbar(bar2, ax=ax2, fraction=.045, pad=.075)
    plt.show()