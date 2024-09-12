import matplotlib.pyplot as plt
import numpy as np

def load_data(train_in_path, train_out_path, test_in_path, fhat_path, fvar_path, size_train: int, size_test: int):
    """Loads training and test data from the specified file paths.

    Args:
        train_in_path (str): Path to the training input data file.
        train_out_path (str): Path to the training output data file.
        test_in_path (str): Path to the test input data file.
        fhat_path (str): Path to the predicted mean values file for the test data.
        fvar_path (str): Path to the predicted variance values file for the test data.
        size_train (int): Number of training samples to load.
        size_test (int): Number of test samples to load.

    Returns:
        tuple: A tuple containing the loaded data (X_train, Y_train, X_test, f_hat, f_var).
    """

    X_train = np.loadtxt(train_in_path, dtype='d')[:size_train, None]
    X_test = np.loadtxt(test_in_path, dtype='d')[:size_test, None]

    Y_train = np.loadtxt(train_out_path, dtype='d')[:size_train, None]
    f_hat = np.loadtxt(fhat_path, dtype='d')[:size_test, None]
    f_var = np.loadtxt(fvar_path, dtype='d')[:size_test, None]

    return X_train, Y_train, X_test, f_hat, f_var

def compute_confidence_interval(f_mean, f_var):
    f_lower = f_mean - 1.96 * np.sqrt(f_var)
    f_upper = f_mean + 1.96 * np.sqrt(f_var)
    return f_lower, f_upper

def plot_process(X_train, Y_train, X_test, f_hat, f_lower, f_upper):
    plt.plot(X_train, Y_train, "kx", mew=2, label="input data")
    plt.plot(X_test, f_hat, "-", color="C0", label="mean")
    plt.plot(X_test, f_lower, "--", color="C0", label="f 95% confidence")
    plt.plot(X_test, f_upper, "--", color="C0")
    plt.fill_between(
        X_test[:, 0], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.1
    )
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    
    train_in_file = "src/data/training/training_input.txt"
    train_out_file = "src/data/training/training_output.txt"
    test_in_file = "src/data/test/test_input.txt"
    fhat_file = "build/predictions.txt"
    fvar_file = "build/uncertainty.txt"

    size_train = 300
    size_test = 700

    X_train, Y_train, X_test, f_hat, f_var = load_data(
        train_in_file,
        train_out_file,
        test_in_file,
        fhat_file,
        fvar_file,
        size_train,
        size_test
        )
    
    f_lower, f_upper = compute_confidence_interval(f_hat, f_var)
    
    train_indices_f = "src/data/smooth/training_indices.txt"
    test_indices_f = "src/data/smooth/test_indices.txt"

    # X_train = np.loadtxt(train_in_file, dtype='d').reshape(300, 1)
    X = np.loadtxt(train_indices_f, dtype='d').reshape(300, 1)
    #X_test = np.loadtxt(test_in_file, dtype='d').reshape(700, 1)
    X_plot = np.loadtxt(test_indices_f, dtype='d').reshape(700, 1)
    
    plot_process(X, Y_train, X_plot, f_hat, f_lower, f_upper)


################################################################################


# import numpy as np

# # Given array of numbers
# Y = [0.        , 0.02040816, 0.04081633, 0.06122449, 0.08163265,
#        0.10204082, 0.12244898, 0.14285714, 0.16326531, 0.18367347,
#        0.20408163, 0.2244898 , 0.24489796, 0.26530612, 0.28571429,
#        0.30612245, 0.32653061, 0.34693878, 0.36734694, 0.3877551 ,
#        0.40816327, 0.42857143, 0.44897959, 0.46938776, 0.48979592,
#        0.51020408, 0.53061224, 0.55102041, 0.57142857, 0.59183673,
#        0.6122449 , 0.63265306, 0.65306122, 0.67346939, 0.69387755,
#        0.71428571, 0.73469388, 0.75510204, 0.7755102 , 0.79591837,
#        0.81632653, 0.83673469, 0.85714286, 0.87755102, 0.89795918,
#        0.91836735, 0.93877551, 0.95918367, 0.97959184, 1.        ]

# # Convert the list to a NumPy array
# numbers_array = np.array(Y)

# # Save the array to a .txt file
# np.savetxt('test_output.txt', numbers_array, fmt='%.8f')

# print("Numbers have been written to numbers_column.txt")



# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Load the data from the .txt files
# #x = np.loadtxt('src/data/smooth/training_input.txt')
# # y = np.loadtxt('src/data/smooth/training_output.txt')
# x = np.loadtxt('/home/maksim/simtech/thesis/msd_simulator/data/training_input.txt')
# y = np.loadtxt('/home/maksim/simtech/thesis/msd_simulator/data/training_output.txt')

# # Ensure that the length of x and y are the same
# assert len(x) == len(y), "The files must have the same number of rows."

# # Determine the number of samples for the subsample (30%)
# n = len(x)
# subsample_size = int(n * 0.30)

# # Generate ordered indices
# indices = np.arange(n)

# # Randomly select subsample indices without replacement
# subsample_indices = np.sort(np.random.choice(indices, size=subsample_size, replace=False))

# # Determine the remaining indices
# remaining_indices = np.setdiff1d(indices, subsample_indices)

# # Split the data based on these indices
# x_subsample = x[subsample_indices]
# y_subsample = y[subsample_indices]

# x_remaining = x[remaining_indices]
# y_remaining = y[remaining_indices]

# # Save the subsample to files x.txt and y.txt
# np.savetxt('src/data/training/training_input.txt', x_subsample, fmt='%f')
# np.savetxt('src/data/training/training_output.txt', y_subsample, fmt='%f')
# np.savetxt('src/data/smooth/training_indices.txt', subsample_indices, fmt='%d')

# # Save the rest of the data to files x1.txt and y1.txt
# np.savetxt('src/data/test/test_input.txt', x_remaining, fmt='%f')
# np.savetxt('src/data/test/test_output.txt', y_remaining, fmt='%f')
# np.savetxt('src/data/smooth/test_indices.txt', remaining_indices, fmt='%d')
