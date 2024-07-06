import sys
import time
import os

sys.path.append(os.path.abspath("../plot"))

import logging
from csv import writer
from config import get_config
from hpx_logger import setup_logging

import gaussian_process as a
from plotting import *
import numpy as np

logger = logging.getLogger()
log_filename = "./hpx_logs.log"


if __name__ == "__main__":

    print(sys.argv)

    n_train = 300
    n_test = 700
    n_tiles = 10

    try:
        n_tile_size = a.compute_train_tile_size(n_train, n_tiles)
        print("Train Tiles:", n_tiles)
    except RuntimeError as e:
        print(e)

    print("Number of training tiles: ", n_tiles)
    m_tiles, m_tile_size = a.compute_test_tiles(n_test, n_tiles, n_tile_size)
    print(f"m_tiles: {m_tiles}, m_tile_size: {m_tile_size}")

    print(f"# Train samples: {n_train}")
    print(f"# Test samples: {n_test}")
    print(f"Tile size in N dimension: {n_tile_size}")
    print(f"Tile size in M dimension: {m_tile_size}")
    print(f"Tiles in N dimension: {n_tiles}")
    print(f"Tiles in M dimension: {m_tiles}")

    hpar = a.Hyperparameters(learning_rate=0.1, opt_iter=0, v_T=[0, 0, 0])
    print(hpar.beta1)
    print(hpar)

    f1_path = (
        "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/training/training_input.txt"
    )
    train_in = a.GP_data(f1_path, n_train)
    print("traning input")
    a.print(train_in.data, 0, 5, ", ")
    f2_path = (
        "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/training/training_output.txt"
    )
    train_out = a.GP_data(f2_path, n_train)
    f3_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/test/test_input.txt"
    test_in = a.GP_data(f3_path, n_test)
    try:
        data = train_in.data
        # print("Data:", data)

        # Call print_slice function
        # Example 1: Print the entire vector
        a.print(data, 0, 3)
        print("data path:", train_in.file_path)
        print("number samples:", train_in.n_samples)

        # Example 2: Print elements from index 1 to 3 (excluding 3)
        # a.print_slice(data, slice(1, 3), ", ")
    except AttributeError as e:
        print(f"Error accessing train_data.data: {e}")

    ###### GP object ######
    gp = a.GP(
        train_in.data,
        train_out.data,
        n_tiles,
        n_tile_size,
        trainable=[False, False, True],
    )
    print("traning input")
    a.print(gp.get_input_data(), 0, 5, ", ")
    print(gp)
    start = time.time()
    a.start_hpx(sys.argv, 2)
    end = time.time()
    print("hpx start time: ", end - start)

    print("Initial loss", gp.compute_loss())

    # for i in range(3):
        # loss = gp.optimize_step(hpar, i)
        # print(f"iter: {i}, loss: {loss}")

    losses = gp.optimize(hpar)
    print("loss")
    a.print(losses)
    pr_and_uncert = gp.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size)
    # print("predict: ")
    # a.print(pr_and_uncert[0])
    print("uncertainty: ")
    a.print(pr_and_uncert[1])
    print(gp)

    pr = gp.predict(test_in.data, m_tiles, m_tile_size)
    print("predict: ")
    # a.print(pr)

    s = time.time()
    a.stop_hpx()
    e = time.time()
    print("hpx shutdown time: ", e - s)

    X = np.array(gp.get_input_data()).reshape(300, 1)
    Y_train = np.array(gp.get_output_data()).reshape(300, 1)
    f_lower, f_upper = compute_confidence_interval(
        np.array(pr_and_uncert[0]).reshape(700, 1), np.array(pr_and_uncert[1]).reshape(700, 1)
    )

    train_indices_f = (
        "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/smooth/training_indices.txt"
    )
    test_indices_f = (
        "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/smooth/test_indices.txt"
    )

    # X_train = np.loadtxt(train_in_file, dtype='d').reshape(300, 1)
    X = np.loadtxt(train_indices_f, dtype="d").reshape(300, 1)
    # X_test = np.loadtxt(test_in_file, dtype='d').reshape(700, 1)
    X_plot = np.loadtxt(test_indices_f, dtype="d").reshape(700, 1)

    plot_process(X, Y_train, X_plot, np.array(pr_and_uncert[0]), f_lower, f_upper)