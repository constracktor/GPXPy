import time
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import gpflow

from scipy import linalg

tf.keras.config.set_floatx('float32')

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
    x_padded = np.pad(
        x_original, pad_width=(n_regressors-1, 0), mode="constant", constant_values=(0)
    )
            
    for _ in range(len(x_original)):
        X.append(x_padded[:n_regressors])
        x_padded = np.roll(x_padded, -1)

    X = np.array(X)

    return X


def load_data(
    train_in_path,
    train_out_path,
    test_in_path,
    test_out_path,
    size_train: int,
    size_test: int,
    n_regressors: int,
):
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

    x_train_in = np.loadtxt(train_in_path, dtype='f')[:size_train]
    x_test_in = np.loadtxt(test_in_path, dtype='f')[:size_test]

    X_train = generate_regressor(x_train_in, n_regressors).astype('f')
    X_test = generate_regressor(x_test_in, n_regressors).astype('f')

    print(X_train.dtype)

    Y_train = np.loadtxt(train_out_path, dtype='f')[:size_train, None]
    Y_test = np.loadtxt(test_out_path, dtype='f')[:size_test, None]

    return X_train, Y_train, X_test, Y_test


def train(X, Y, params_summary: bool = False):
    """_summary_

    Args:
        X (_type_): _description_
        Y (_type_): _description_

    Returns:
        _type_: _description_
    """
    model = gpflow.models.GPR(
        (X, Y),
        kernel= gpflow.kernels.SquaredExponential(variance=1.0, lengthscales=1.0), # gpflow.kernels.White(0.1) + 
        # mean_function = gpflow.functions.Constant(),
        # likelihood = gpflow.likelihoods.Gaussian(0.1),
        noise_variance = 0.1,
    )
    
    if params_summary:
        gpflow.utilities.print_summary(model)

    print(model.trainable_variables)
    # opt = gpflow.optimizers.Scipy()
    # opt.minimize(model.training_loss, model.trainable_variables)

    return model


def predict(model, X_test):
    """_summary_

    Args:
        model (_type_): _description_
        X_test (_type_): _description_

    Returns:
        _type_: _description_
    """
    f_pred, f_var = model.predict_f(X_test)
    y_pred, y_var = model.predict_y(X_test)
    gpflow.utilities.print_summary(model)
    return f_pred, f_var, y_pred, y_var


if __name__ == "__main__":

    train_in_file = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/training/training_input.txt"
    train_out_file = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/training/training_output.txt"
    test_in_file = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/test/test_input.txt"
    test_out_file = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/test/test_output.txt"

    size_train = 10
    size_test = 100
    n_regressors = 10

    X_train, Y_train, X_test, Y_test = load_data(
        train_in_file, train_out_file,
        test_in_file, test_out_file,
        size_train, size_test,
        n_regressors,
    )

    # print(tf.config.threading.get_inter_op_parallelism_threads())
    # print(tf.config.threading.get_intra_op_parallelism_threads())
    # num_threads = 6  # check lscpu to identify number of threads
    # tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    # tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    gpflow.config.set_default_float(np.float32)

    gp_model = train(X_train, Y_train)

    k_matrix = gp_model.kernel(X_train).numpy()
    # print(k_matrix)
    hpx_pred_file = "/home/maksim/simtech/thesis/GPPPy_hpx/build/covariance.txt"
    hpx_cov = np.loadtxt(hpx_pred_file, dtype=float)
    # print(hpx_cov.reshape((10,100)))

    cross_cov = gp_model.kernel(X_test, X_train).numpy()
    # print(cross_cov.reshape((10,10)))
    # print(cross_cov.shape)
    print("---------")
    
    # x = linalg.solve(k_matrix, Y_train, assume_a='sym')
    
    # manual_pred = np.dot(cross_cov, x)
    
    # print(manual_pred[:5])
    
    print("+++++++++++++")
    t_cross_cov = gp_model.kernel(X_train, X_test).numpy()
    # print(t_cross_cov
    # print(t_cross_cov.shape)
        
    # Z = linalg.solve(k_matrix, t_cross_cov)
    # # print(Z.flatten()[:28])
    # # print(Z.shape)
    prior_k = gp_model.kernel(X_test).numpy()
    # print(prior_k)
    # posterior_k = prior_k - np.dot(cross_cov, Z)
    # var = np.diag(posterior_k)
    # print(var[:5])
    # print("+++++++++++++++")
    # print(X_test.dtype)
    f_pred, f_var, y_pred, y_var = predict(gp_model, X_test)
    # print(Y_test[:5])
    print(f_pred[-5:])
    # print(y_pred[:5])

    hpx_pred_file = "/home/maksim/simtech/thesis/GPPPy_hpx/build/predictions.txt"
    hpx_pred = np.loadtxt(hpx_pred_file, dtype=float)[:size_test, None]
    print(np.array_equal(hpx_pred.shape, f_pred.numpy()))

    print(f_var.numpy()[-5:])
    print(hpx_pred[-5:])

    # test_in_file = "src/data/test/test_input.txt"
    # Xplot = np.loadtxt(test_in_file, dtype=float)[:100, None]
    # Yplot = np.loadtxt("src/data/test/test_output.txt", dtype=float)[:100, None]
    # plt.plot(Xplot, Yplot, "kx", mew=2)
    # plt.show()

    # print(Xplot.shape)
    # f_mean, f_var = model.predict_f(Xplot, full_cov=False)
    # y_mean, y_var = model.predict_y(Xplot)

    # f_lower = f_mean - 1.96 * np.sqrt(f_var)
    # f_upper = f_mean + 1.96 * np.sqrt(f_var)
    # y_lower = y_mean - 1.96 * np.sqrt(y_var)
    # y_upper = y_mean + 1.96 * np.sqrt(y_var)

    # plt.plot(X, Y, "kx", mew=2, label="input data")
    # plt.plot(Xplot, f_mean, "-", color="C0", label="mean")
    # plt.plot(Xplot, f_lower, "--", color="C0", label="f 95% confidence")
    # plt.plot(Xplot, f_upper, "--", color="C0")
    # plt.fill_between(
    #     Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.1
    # )
    # plt.plot(Xplot, y_lower, ".", color="C0", label="Y 95% confidence")
    # plt.plot(Xplot, y_upper, ".", color="C0")
    # plt.fill_between(
    #     Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1
    # )
    # plt.legend()
    # plt.show()

    # std::ofstream input_filee("./input.txt");
    # std::ostream_iterator<CALC_TYPE> input_iterator(input_filee, "\n");
    # std::copy(test_output.begin(), test_output.end(), input_iterator);


    '''
    // Regressor generation in Lexi Code
    
    #include <iostream>

    void regress(int i_global, int j_global, int n_regressors) {
    for (int k = 0; k < n_regressors; k++)
    {
        //
        int offset = -n_regressors + 1 + k;
        int i_local = i_global + offset;
        int j_local = j_global + offset;
        //
        std::cout << i_global << "." << j_global << ", " << i_local << "." << j_local << ": ";
        std::cout << -n_regressors + 1 + k << "\n" ;
    }
    }


    int main() {
        // Write C++ code here
        int n_regressors = 5;
        int n_tiles = 2;
        int N = 2;
        
        for (std::size_t row = 0; row < n_tiles; row++)
        {
            for (std::size_t col = 0; col <= row; col++)
            {
                std::size_t i_global, j_global;
                for (std::size_t i = 0; i < N; i++)
                {
                    i_global = N * row + i;
                    for (std::size_t j = 0; j < N; j++)
                    {
                    j_global = N * col + j;
                    regress(i_global, j_global, n_regressors);
                    }
                }
            }
        }
        return 0;
    }
    '''