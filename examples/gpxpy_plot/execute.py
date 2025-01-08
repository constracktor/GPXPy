import sys

from config import get_config
import gpxpy as gpx

import matplotlib.pyplot as plt

config = get_config()

n_train = config["N_TRAIN"]
n_tile_size = gpx.compute_train_tile_size(n_train, config["N_TILES"])
train_in = gpx.GP_data(config["train_in_file"], n_train)
train_out = gpx.GP_data(config["train_out_file"], n_train)

m_tiles, m_tile_size = gpx.compute_test_tiles(config["N_TEST"], config["N_TILES"], n_tile_size)
test_in = gpx.GP_data(config["test_in_file"], config["N_TEST"])

gpx.start_hpx(sys.argv, 8)

gp = gpx.GP(train_in.data, train_out.data, config["N_TILES"], n_tile_size,
            lengthscale=1.0, v_lengthscale=1.0 , noise_var=0.2, n_reg=1, trainable=[True, True, True])

hpar = gpx.AdamParams(learning_rate=0.1, opt_iter=10, m_T=[0,0,0], v_T=[0,0,0])
losses = gp.optimize(hpar)

pr, var = gp.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size)

# pr_ = gp.predict(test_in.data, m_tiles, m_tile_size)
# pr__, var__ = gp.predict_with_full_cov(test_in.data, m_tiles, m_tile_size)

gpx.stop_hpx()

# plot gp pr, var, data (note that var is twice as long as pr), use plt
plt.plot(test_in.data, pr)
plt.scatter(train_in.data, train_out.data)
pr_plus_var = [p + v for p, v in zip(pr, var)]
pr_minus_var = [p - v for p, v in zip(pr, var)]
plt.plot(test_in.data, pr_plus_var, 'r')
plt.plot(test_in.data, pr_minus_var, 'r')
plt.savefig("plot.png")

