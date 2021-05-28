import numpy as np
from matplotlib import pyplot as plt
import random
import csv

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT:
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        row_num = 0
        for row in reader:
            row_num += 1
            if row_num == 1:
                continue
            del row[0]
            for i in range(len(row)):
                row[i] = float(row[i])
            dataset.append(row)
    return np.array(dataset)



def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on.
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    num = dataset.shape[0]
    print(num)
    
    mean = (np.sum(dataset.transpose()[col]))/num
    print('{:.2f}'.format(mean))
    
    total = 0
    for i in range (num):
        total += (dataset[i][col] - mean) ** 2
    sd = (total/(num - 1)) ** 0.5
    print('{:.2f}'.format(sd))
    pass


def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    mse = None
    n = len(dataset)
    
    squared_errors = 0
    for row in dataset:
        error_sum = 0
        error_sum += betas[0]
        for i in range(1, len(betas)):
            error_sum += betas[i]*row[cols[i-1]]
        error_sum -= row[0]
        squared_errors += error_sum**2
    mse = (1/n) * squared_errors
    return mse


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    grads = []
    for i in range(len(betas)):
        entry = 0
        for j in range(len(dataset)):
            temp = betas[0] + np.dot(dataset[j][cols],betas[1:]) - dataset[j][0]
            if i == 0:
                entry += temp
            else:
                entry += temp * dataset[j][cols[i-1]]
        grads.append(entry*2/len(dataset))       
    return np.array(grads)



def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    for i in range(T):
        curr_betas = gradient_descent(dataset, cols, betas)
        for j in range(len(betas)):
            betas[j] = betas[j] - eta * curr_betas[j]
        print(i + 1, '{:.2f}'.format(regression(dataset, cols, betas)),
              *['{:.2f}'.format(curr_b) for curr_b in betas])


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    xset = []
    yset = dataset[:, 0]
    for i in range(len(dataset)):
        xset.append([1])
    for j in range(len(dataset)):
        for k in range(len(cols)):
            xset[j].append(dataset[j][cols[k]])
    x_transpose = np.transpose(np.array(xset))
    bet_output = np.dot(np.linalg.inv(np.dot(x_transpose, np.array(xset))), np.dot(x_transpose, np.array(yset)))
    return regression(dataset, cols, bet_output), *bet_output

def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = compute_betas(dataset, cols)
    return betas[1] + np.sum(np.multiply(betas[2:], features))


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    mu = 0
    z = np.random.normal(mu, sigma, len(X))
    
    b_x = betas[1] * X[:, 0]
    b_0 = np.asarray(betas[0])
    linear_output = b_0 + b_x + z
    linear_output = np.column_stack((linear_output, X[:, 0]))

    a_x = alphas[1] * np.square(X[:, 0])
    a_0 = np.asarray(alphas[0])
    quadratic_output = a_0 + a_x + z
    quadratic_output = np.column_stack((quadratic_output, X[:, 0]))
    
    return linear_output, quadratic_output



def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    X = []
    dataset = []
    lin = []
    quad = []
    for i in range(1000):
        X.append(random.randint(-100, 100))
    X = np.reshape(X, (1000, 1))
    betas = [1, 2]
    alphas = [3, 4]
    sigmas = [10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3, 10**4, 10**5]
    for i in range(len(sigmas)):
        lin_out, quad_out = synthetic_datasets(betas, alphas, X, sigmas[i])
        dataset.append((lin_out, quad_out))
    for i in range(len(sigmas)):
        lin_beta = compute_betas(dataset[i][0], cols=[1])
        quad_beta = compute_betas(dataset[i][1], cols=[1])
        lin.append(lin_beta[0])
        quad.append(quad_beta[0])
        
    plt.plot(sigmas, lin, marker='o', label='Linear')
    plt.plot(sigmas, quad, marker='o', label='Quadratic')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Sigma")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("mse.pdf")


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()