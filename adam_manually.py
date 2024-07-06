# gradient descent optimization with adam for a two-dimensional test function
from math import sqrt
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import  cholesky, solve_triangular

def softplus(x):
    return np.log(1+np.exp(x))

def der_sofplus(x):
    return 1/(np.exp(-x)+1)

def inv_softplus(x):
    return np.log(np.exp(x)-1)

def der_inv_sofplus(x):
    return np.exp(x)/(np.exp(x)-1)

def sigmoid(x):
    return 1. / (1.+ np.exp(-x))

def inv_sigmoid(x):
    return np.log(x / (1-x))

def der_sigmoid(x):
    return sigmoid(x)*(1. - sigmoid(x))

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
    x_padded = np.pad(x_original, pad_width=(n_regressors, 0), mode="constant", constant_values=(0))
    for _ in range(len(x_original)):
        x_padded = np.roll(x_padded, -1)
        X.append(x_padded[:n_regressors])

    X = np.array(X, dtype='d')

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

    x_train_in = np.loadtxt(train_in_path, dtype='d')[:size_train]
    x_test_in = np.loadtxt(test_in_path, dtype='d')[:size_test]

    X_train = generate_regressor(x_train_in, n_regressors)
    X_test = generate_regressor(x_test_in, n_regressors)

    Y_train = np.loadtxt(train_out_path, dtype='d')[:size_train, None]
    Y_test = np.loadtxt(test_out_path, dtype='d')[:size_test, None]

    return X_train, Y_train, X_test, Y_test

def to_constrained(parameter):
    # transform hyperparameter to enforce constraints using softplus
    return np.log(1.0 + np.exp(parameter)) + 1e-6
    # return log(1.0 + exp(parameter));

def to_unconstrained(parameter):
    # transform hyperparmeter to entire real line using inverse 
    # of sofplus. Optimizer, such as gradient decent or Adam, 
    # work better with unconstrained parameters
    return np.log(np.exp(parameter - 1e-6) - 1.0)
    # return log(exp(parameter) - 1.0);

def rhs(X_train, Y_train, hyperparam):
    K = rbf_kernel(X_train, gamma=1./(2*1.0**2)) + hyperparam*np.eye(X_train.shape[0])
        
    # print(np.all(np.linalg.eigvals(K) > 0))
    # print(np.linalg.cond(K)
    L = cholesky(K, lower=True)
    # print(solve_triangular(L.T, solve_triangular(L, np.eye(X_train.shape[0]), lower=True)) @ Y_train)
    
    alpha = solve_triangular(L, Y_train, lower=True)
    alpha = solve_triangular(L.T, alpha)
    return K, L, alpha

def pred_and_uncert(X_train, Y_train, X_test):
    K = rbf_kernel(X_train, gamma=1./(2*1.0**2)) + 0.1*np.eye(X_train.shape[0])   
    priorK = rbf_kernel(X_test, gamma=1./(2*1.0**2))
    crossK = rbf_kernel(X_test, X_train, gamma=1./(2*1.0**2))
    t_crossK = rbf_kernel(X_train, X_test, gamma=1./(2*1.0**2))
    # print(np.all(np.linalg.eigvals(K) > 0))
    # print(np.linalg.cond(K)
    L = cholesky(K, lower=True)
    pred = crossK @ solve_triangular(L.T, solve_triangular(L, Y_train, lower=True))
    V = solve_triangular(L, t_crossK, lower=True)
    print(V.T[0])
    print(np.diag(V.T @ V))
    uncertainty = np.diag(priorK) - np.diag(V.T @ V)
    return pred, uncertainty

# objective function
def objective(X_train, Y_train, L, alpha):
    N = X_train.shape[0]
    loss =  np.sum(np.log(np.diag(L)**2, dtype='d')) + Y_train.T.dot(alpha) + N * np.log(2*np.pi, dtype='d')
    return 0.5*loss/N

def compute_distance_matrix(X, Y):
    # Compute the pairwise distances using broadcasting
    return np.sum((X[:, np.newaxis] - Y[np.newaxis, :])**2, axis=2)

# derivative of objective function
def derivative(X_train, Y_train, K, alpha, hyperparam):
    N = X_train.shape[0]
    grad_noise = np.eye(X_train.shape[0], X_train.shape[0]) * sigmoid(inv_softplus(hyperparam - 1.e-6))
    # print("grad left ", np.trace(np.linalg.solve(K, (np.eye(X_train.shape[0]))) @ grad_noise))
    # print(np.linalg.solve(K, (np.eye(X_train.shape[0]))))
    # print(np.dot(alpha.T, np.dot(grad_noise, alpha))[0][0])
    return 0.5 * 1/N * np.trace(np.linalg.solve(K, (np.eye(X_train.shape[0]))) @ grad_noise) - 0.5 * 1/N * np.dot(alpha.T, np.dot(grad_noise, alpha))[0][0]
    # return 0.5 * 1/N * np.trace( np.linalg.solve(K, (np.eye(X_train.shape[0]) - np.outer(Y_train, alpha))) @  grad_noise )
    
    # grad_v = rbf_kernel(X_train, gamma=0.5) * sigmoid(inv_softplus(hyperparam))
    # print(np.diag(np.linalg.solve(K, (np.eye(X_train.shape[0]))) @ grad_v))
    # print(np.trace(np.linalg.solve(K, (np.eye(X_train.shape[0]))) @ grad_v))
    # grad_left = 0.5 * 1/N * np.trace(np.linalg.solve(K, (np.eye(X_train.shape[0]))) @ grad_v)
    # print(grad_left)
    # grad_right = - 0.5 * 1/N * np.dot(alpha.T, np.dot(grad_v, alpha))
    # print(grad_right[0][0])
    # # return 0.5 * 1/N * np.trace( np.linalg.solve(K, (np.eye(X_train.shape[0]) - np.outer(Y_train, alpha))) @  grad_v )
    # return grad_left + grad_right[0][0]
    # grad_l = 1.0/ (hyperparam**3) * compute_distance_matrix(X_train, X_train) * rbf_kernel(X_train, gamma=1./(2*(hyperparam**2))) * sigmoid(inv_softplus(hyperparam))
    # return 0.5 * 1/N * np.trace( np.linalg.solve(K, (np.eye(X_train.shape[0]) - np.outer(Y_train, alpha))) @  grad_l  )

# gradient descent algorithm with adam
def adam(objective, derivative, X_train, Y_train, noise, n_iter, lr, beta1, beta2, eps=1e-8):
    # generate an initial point
    K, L, a = rhs(X_train, Y_train, noise)
    score = objective(X_train, Y_train, L, a)
    # print(f'initial f({noise:.6f}) = {score[0][0]:.6f}')
    # initialize first and second moments
    m = [0.0]
    v = [0.0]
    # run the gradient descent updates
    for t in range(n_iter):
        
        # print(f'>{t} f({noise:.10f}) = {score[0][0]:.10f}')
        print(f'>{t} loss: {score[0][0]:.17f}, param: {noise:.17f}')
        # calculate gradient g(t)
        g = derivative(X_train, Y_train, K, a, noise)
        print(f'gradient: {g:.12f}')
        # build a solution one variable at a time
        for i in range(1):
            # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
            m[i] = beta1 * m[i] + (1.0 - beta1) * g
            # print("m", m[i], ", i: ", i)
            # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
            v[i] = beta2 * v[i] + (1.0 - beta2) * g**2
            # print("v", v[i])
            # mhat(t) = m(t) / (1 - beta1(t))
            # print("beta1 ", beta1**(t+1))
            mhat = m[i] / (1.0 - beta1**(t+1))
            # print("mhat ", mhat)
            # print("mhat ", mhat)
            # vhat(t) = v(t) / (1 - beta2(t))
            vhat = v[i] / (1.0 - beta2**(t+1))
            # print(vhat)
            # print("vhat ", vhat)
            # # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
            # print("change ", alpha * mhat / (sqrt(vhat) + eps))
            # print("to_unconstrained(noise) ", to_unconstrained(noise))
            alpha_T = lr * np.sqrt(1.0 - beta2**(t+1)) / (1.0 - beta1**(t+1))
            # print("alpha_T", alpha_T)
            # print("unconstrained ", to_unconstrained(noise))
            # print("back 1 ", - lr * mhat )
            # print("back 2 ", (sqrt(vhat) + eps) )
            # print("back part ", - lr * mhat / (sqrt(vhat) + eps) )
            # print("before_constr a ",  (lr * mhat) / (sqrt(vhat) + eps))
            # print("before_constr aT ",  (alpha_T * m[i]) / (np.sqrt(v[i], dtype='d') + eps))
            noise = to_constrained(to_unconstrained(noise) - alpha_T * m[i] / (np.sqrt(v[i], dtype='d') + eps))
            # print(" noise alphaT", to_constrained(to_unconstrained(noise) - lr * mhat / (sqrt(vhat) + eps)) )
            # noise = to_constrained(to_unconstrained(noise) - lr * mhat / (sqrt(vhat) + eps))
        # evaluate candidate point
        K, L, a = rhs(X_train, Y_train, noise)
        # for elem in K:
        #     for e in elem:
        #         print(f'{e:.12f}')
        score = objective(X_train, Y_train, L, a)
        # report progress
        # if t % 10 == 0:
        # print(f'>{t} f({noise:.7f}) = {score[0][0]:.6f}')
    return [noise, score[0][0]]


if __name__ == "__main__":
    # seed the pseudo random number generator
    np.random.seed(7)
    # define range for input
    # bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
    # define the total iterations
    train_in_file = "src/data/training/training_input.txt"
    train_out_file = "src/data/training/training_output.txt"
    test_in_file = "src/data/test/test_input.txt"
    test_out_file = "src/data/test/test_output.txt"

    size_train = 80
    size_test = 50
    n_regressors = 100

    X_train, Y_train, X_test, Y_test = load_data(
        train_in_file,
        train_out_file,
        test_in_file,
        test_out_file,
        size_train,
        size_test,
        n_regressors,
    )
    
    noise = 0.1
    # number of optimisation steps
    n_iter = 0
    # steps size
    lr = 0.1
    # factor for average gradient
    beta1 = 0.9
    # factor for average squared gradient
    beta2 = 0.999
    # perform the gradient descent search with adam
    pred, uncert = pred_and_uncert(X_train, Y_train, X_test)
    print(uncert)
    best, score = adam(objective, derivative, X_train, Y_train, noise, n_iter, lr, beta1, beta2)
    print('------')
    print(f'f({best:.14f}) = {score:.10f}')



# iter: 0 loss: 1.263018328416 param: 0.200000000000
# gradient: 0.082484334523
# m_T: 0.008248433452
# v_T: 0.000006803665
# alpha_T: 0.000316227766
# unconstrained_hyperparam: -1.507777317638544
# back part: 0.082484334523
# before constr : -1.508777317638544
# iter: 1 loss: 1.262935854803 param: 0.199818805761
# gradient: 0.082462890009
# m_T: 0.015667949102
# v_T: 0.000013593456
# alpha_T: 0.000235316725
# unconstrained_hyperparam: -1.508777317638544
# back part: -0.001000000000
# before constr : -1.509777317638544

