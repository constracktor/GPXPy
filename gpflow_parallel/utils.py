import numpy as np
import gpflow


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
        x_original,
        pad_width=(n_regressors - 1, 0),
        mode="constant",
        constant_values=(0),
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

    x_train_in = np.loadtxt(train_in_path, dtype="f")[:size_train]
    x_test_in = np.loadtxt(test_in_path, dtype="f")[:size_test]

    X_train = generate_regressor(x_train_in, n_regressors).astype("f")
    X_test = generate_regressor(x_test_in, n_regressors).astype("f")

    Y_train = np.loadtxt(train_out_path, dtype="f")[:size_train, None]
    Y_test = np.loadtxt(test_out_path, dtype="f")[:size_test, None]

    return X_train, Y_train, X_test, Y_test


def train(X, Y, k_var=1.0, k_lscale=1.0,  noise_var=0.1, params_summary: bool = False):
    """_summary_

    Args:
        X (_type_): _description_
        Y (_type_): _description_

    Returns:
        _type_: _description_
    """
    model = gpflow.models.GPR(
        (X, Y),
        kernel=gpflow.kernels.SquaredExponential(
            variance=k_var,
            lengthscales=k_lscale,
        ),
        # likelihood = gpflow.likelihoods.Gaussian(0.1),
        noise_variance=noise_var,
    )

    if params_summary:
        gpflow.utilities.print_summary(model)

    return model


def optimize_model(model):
    # opt = gpflow.optimizers.Scipy()
    # opt.minimize(model.training_loss, model.trainable_variables)
    return None


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
    
    return f_pred, f_var, y_pred, y_var


def calculate_error(Y_test, Y_pred):
    """_summary_

    Args:
        Y_test (_type_): _description_
        Y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.linalg.norm(Y_test - Y_pred)