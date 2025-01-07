import sys

from config import get_config

print("importing gpx")
import gpxpy as gpx
print("imported gpx")

import matplotlib.pyplot as plt
def gpx_run(config, n_train, cores):

    n_tile_size = gpx.compute_train_tile_size(n_train, config["N_TILES"])
    m_tiles, m_tile_size = gpx.compute_test_tiles(config["N_TEST"], config["N_TILES"], n_tile_size)
    hpar = gpx.AdamParams(learning_rate=0.1, opt_iter=config["OPT_ITER"], m_T=[0,0,0], v_T=[0,0,0])
    train_in = gpx.GP_data(config["train_in_file"], n_train)
    train_out = gpx.GP_data(config["train_out_file"], n_train)
    test_in = gpx.GP_data(config["test_in_file"], config["N_TEST"])

    gpx.start_hpx(sys.argv, cores)

    gp = gpx.GP(train_in.data, train_out.data, config["N_TILES"], n_tile_size, trainable=[True, True, True])

    print("created gp")

    losses = gp.optimize(hpar)

    print("optimized gp")

    pr, var = gp.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size)

    print("predicted gp")

    # pr__, var__ = gp.predict_with_full_cov(test_in.data, m_tiles, m_tile_size)
    # pr_ = gp_cpu.predict(test_in.data, m_tiles, m_tile_size)

    # plot gp pr, var, data
    plt.figure()
    plt.plot(test_in.data, pr, label='pr')
    plt.plot(test_in.data, pr + 2 * var, label='pr + 2*var')
    plt.plot(test_in.data, pr - 2 * var, label='pr - 2*var')
    plt.plot(train_in.data, train_out.data, 'ro', label='data')
    plt.legend()
    plt.show()


    gpx.stop_hpx()

if __name__ == "__main__":
    config = get_config()
    data_size = config["END"]
    cores = config["N_CORES"]

    gpx_run(config, data_size, cores)
