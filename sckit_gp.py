import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from scipy import linalg

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

def generate_regressor(x_original, n_regressors):
    """Pad X with zeros from left for #regressors-1 for each data point
    roll over X where window size equals #regressors

    Args:
        x_original (_type_): _description_
        n_regressors (_type_): _description_

    Returns:
        _type_: _description_
    """
    X = []
    x_padded = np.pad(x_original, pad_width=(n_regressors-1, 0), mode="constant", constant_values=(0))
    for _ in range(len(x_original)):
        X.append(x_padded[:n_regressors])
        x_padded = np.roll(x_padded, -1)

    X = np.array(X)

    return X


def load_data(train_in_path, train_out_path, test_in_path, test_out_path, size_train: int, size_test: int, n_regressors: int):
    """_summary_

    Args:
        train_in_path (_type_): _description_
        train_out_path (_type_): _description_
        test_in_path (_type_): _description_
        test_out_path (_type_): _description_
        size (int): _description_
        n_regressors (int): _description_

    Returns:
        _type_: _description_
    """

    x_train_in = np.loadtxt(train_in_path, dtype=float)[:size_train]
    x_test_in = np.loadtxt(test_in_path, dtype=float)[:size_test]

    X_train = generate_regressor(x_train_in, n_regressors)
    X_test = generate_regressor(x_test_in, n_regressors)

    Y_train = np.loadtxt(train_out_path, dtype=float)[:size_train, None]
    Y_test = np.loadtxt(test_out_path, dtype=float)[:size_test, None]

    return X_train, Y_train, X_test, Y_test

def gpPrediction( l, X_train, y_train, X_test):
  # Kernel definition 
  kernel = RBF(length_scale=l, length_scale_bounds='fixed') # WhiteKernel(noise_level=0.1, noise_level_bounds="fixed") \
  # GP model 
  gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=0.1)
  # Fitting in the gp model
  gp.fit(X_train, y_train)
  # Make the prediction on test set.
  y_pred, sigma = gp.predict(X_test, return_std=True)
  return y_pred, sigma, gp


if __name__ == "__main__":
    
    train_in_file = "src/data/training/training_input.txt"
    train_out_file = "src/data/training/training_output.txt"
    test_in_file = "src/data/test/test_input.txt"
    test_out_file = "src/data/test/test_output.txt"

    size_train = 100
    size_test = 10
    n_regressors = 10

    X_train, Y_train, X_test, Y_test = load_data(
        train_in_file,
        train_out_file,
        test_in_file,
        test_out_file,
        size_train,
        size_test,
        n_regressors,
    )
        
    l_init = 1.0

    y_pred, sigma, gp = gpPrediction( l_init, X_train, Y_train, X_test)
    print(y_pred[:5])
    print((sigma**2)[:5])
    
    print("-----------")
    # k_matrix = gp.kernel_(X_train)
    cross_cov = gp.kernel_(X_test, X_train)
    # print(cross_cov)
    t_cross_cov = gp.kernel_(X_train, X_test)
    # print(t_cross_cov)
    prior_k = gp.kernel_(X_test)
    # print(prior_k)
    Z_l = linalg.solve_triangular(gp.L_, t_cross_cov, lower=True)
    Z_u = linalg.solve(gp.L_, Z_l, transposed=True)
    posterior_k = prior_k - np.dot(cross_cov, Z_u)
    var = np.diag(posterior_k)
    # print(var[:5])

    print("###############")
    
    hpx_pred_file = "/home/maksim/simtech/thesis/GPPPy_hpx/build/predictions.txt"
    hpx_cov = np.loadtxt(hpx_pred_file, dtype=float)
    print(hpx_cov[:5])