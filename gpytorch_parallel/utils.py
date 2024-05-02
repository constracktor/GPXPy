import numpy as np
import gpytorch
import torch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        self.covar_module.base_kernel.lengthscale = 1.0
        self.covar_module.outputscale = 1.0
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

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

    x_train_in = np.loadtxt(train_in_path, dtype=float)[:size_train]
    x_test_in = np.loadtxt(test_in_path, dtype=float)[:size_test]

    X_train = torch.from_numpy(generate_regressor(x_train_in, n_regressors))
    X_test = torch.from_numpy(generate_regressor(x_test_in, n_regressors))

    Y_train = torch.from_numpy(np.loadtxt(train_out_path, dtype=float)[:size_train])
    Y_test = torch.from_numpy(np.loadtxt(test_out_path, dtype=float)[:size_test])

    return X_train, Y_train, X_test, Y_test


def train(model, likelihood, X_train, Y_train, training_iter=10):
    """_summary_

    Args:
        X (_type_): _description_
        Y (_type_): _description_

    Returns:
        _type_: _description_
    """
    # training_iter = 10

    # # Find optimal model hyperparameters
    # model.train()
    # likelihood.train()

    # # Use the adam optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # # "Loss" for GPs - the marginal log likelihood
    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # for i in range(training_iter):
    #     # Zero gradients from previous iteration
    #     optimizer.zero_grad()
    #     # Output from model
    #     output = model(X_train)
    #     # Calc loss and backprop gradients
    #     loss = -(mll(output, Y_train))
    #     loss.backward()
    #     print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
    #         i + 1, training_iter, loss.item(),
    #         model.covar_module.base_kernel.lengthscale.item(),
    #         model.likelihood.noise.item()
    #     ))
    #     optimizer.step()
        
    return None


def predict(model, likelihood, X_test):
    """_summary_

    Args:
        model (_type_): _description_
        X_test (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_pred = model(X_test)
        f_mean = f_pred.mean
        f_var = f_pred.variance
        
        y_pred = likelihood(model(X_test))
        y_mean = y_pred.mean
        y_var = y_pred.variance

    '''
    for future plot generation:
    # Get upper and lower confidence bounds
    observed_pred = likelihood(model(test_x))
    lower, upper = observed_pred.confidence_region()
    '''
    
    return f_mean, f_var, y_mean, y_var


def calculate_error(Y_test, Y_pred):
    """_summary_

    Args:
        Y_test (_type_): _description_
        Y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    return torch.norm(Y_test - Y_pred)