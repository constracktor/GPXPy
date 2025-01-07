import sys

from config import get_config
import gpxpy as gpx

import matplotlib.pyplot as plt

config = get_config()

n_train = config["N_TRAIN"]

n_tile_size = gpx.compute_train_tile_size(n_train, config["N_TILES"])
train_in = gpx.GP_data(config["train_in_file"], n_train)
train_out = gpx.GP_data(config["train_out_file"], n_train)

# m_tiles, m_tile_size = gpx.compute_test_tiles(config["N_TEST"], config["N_TILES"], n_tile_size)
# test_in = gpx.GP_data(config["test_in_file"], config["N_TEST"])
#
# gpx.start_hpx(sys.argv, 8)
#
# gp = gpx.GP(train_in.data, train_out.data, config["N_TILES"], n_tile_size, trainable=[True, True, True])
#
# hpar = gpx.AdamParams(learning_rate=0.1, opt_iter=100, m_T=[0,0,0], v_T=[0,0,0])
# losses = gp.optimize(hpar)
#
# pr, var = gp.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size)
#
# # pr_ = gp.predict(test_in.data, m_tiles, m_tile_size)
# # pr__, var__ = gp.predict_with_full_cov(test_in.data, m_tiles, m_tile_size)
#
# gpx.stop_hpx()

# plot gp pr, var, data (note that var is twice as long as pr), use plt
# plt.scatter(t.data, pr, label="GP prediction")
plt.scatter(train_in.data[:512], train_out.data[:512], label="Training data")
plt.scatter(train_in.data[512:], train_out.data[512:], label="Validation data")
plt.legend()
plt.savefig("plot.png")

