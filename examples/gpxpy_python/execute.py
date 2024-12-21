import sys
import time
import os
sys.path.append(os.path.abspath("../plot"))
import logging
from csv import writer
from config import get_config
from hpx_logger import setup_logging

import gpxpy as gpx

logger = logging.getLogger()
log_filename = "./hpx_logs.log"

def gpx_run(config, output_csv_obj, n_train, l, cores):

    total_t = time.time()

    n_tile_size = gpx.compute_train_tile_size(n_train, config["N_TILES"])
    m_tiles, m_tile_size = gpx.compute_test_tiles(config["N_TEST"], config["N_TILES"], n_tile_size)
    hpar = gpx.AdamParams(learning_rate=0.1, opt_iter=config["OPT_ITER"], m_T=[0,0,0], v_T=[0,0,0])
    train_in = gpx.GP_data(config["train_in_file"], n_train)
    train_out = gpx.GP_data(config["train_out_file"], n_train)
    test_in = gpx.GP_data(config["test_in_file"], config["N_TEST"])

    # Init hpx runtime but do not start it yet
    gpx.start_hpx(sys.argv, cores)

    ###### GP object ######
    init_t = time.time()
    gp_cpu = gpx.GP(train_in.data, train_out.data, config["N_TILES"], n_tile_size, trainable=[True, True, True])
    gp_gpu = gpx.GP(train_in.data, train_out.data, config["N_TILES"], n_tile_size, trainable=[True, True, True], gpu_id=0, n_streams=1)
    init_t = time.time() - init_t

    gpx.suspend_hpx()
    gpx.resume_hpx()

    # Calculate Cholesky decomposition
    chol_t = time.time()
    matrix_cpu = gp_cpu.cholesky()
    matrix_gpu = gp_gpu.cholesky()
    chol_t = time.time() - chol_t

    # Perform optmization
    # chol_t = time.time()
    # losses = gp_cpu.optimize(hpar)
    # chol_t = time.time() - chol_t
    # logger.info("Finished optimization.")

    # # Predict
    # pred_uncer_t = time.time()
    # pr, var = gp_cpu.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size)
    # pred_uncer_t = time.time() - pred_uncer_t
    # logger.info("Finished predictions.")

    # Predict
    # pred_full_t = time.time()
    # pr__, var__ = gp_cpu.predict_with_full_cov(test_in.data, m_tiles, m_tile_size)
    # pred_full_t = time.time() - pred_full_t
    # logger.info("Finished predictions with full cov.")

    # Predict
    # pred_t = time.time()
    # pr_ = gp_cpu.predict(test_in.data, m_tiles, m_tile_size)
    # pred_t = time.time() - pred_t
    # logger.info("Finished predictions.")

    # Stop HPX runtime
    gpx.stop_hpx()

    TOTAL_TIME = time.time() - total_t
    INIT_TIME = init_t
    OPTI_TIME = chol_t
    PRED_UNCER_TIME = 0 # pred_uncer_t
    PRED_FULL_TIME = 0 # pred_full_t
    PREDICTION_TIME = 0 # pred_t

    # config and measurements
    row_data = [cores, n_train, config["N_TEST"], config["N_TILES"], config["N_REG"], config["OPT_ITER"],
                TOTAL_TIME, INIT_TIME, OPTI_TIME, PRED_UNCER_TIME, PRED_FULL_TIME, PREDICTION_TIME, l]
    output_csv_obj.writerow(row_data)

    logger.info("Completed iteration.")

def execute():
    """
    Execute the main process:
    - Set up logging.
    - Load configuration file.
    - Initialize output CSV file.
    - Write header to the output CSV file.
    - Iterate through different training sizes and for each training size
    """

    # setup logging
    setup_logging(log_filename, True, logger)

    # load config
    logger.info("\n")
    logger.info("-" * 40)
    logger.info("Load config file.")
    config = get_config()

    # append log to ./output.csv
    file_exists = os.path.isfile("./output.csv")
    output_file = open("./output.csv", "a", newline="")
    output_csv_obj = writer(output_file)

    # write headers
    if not file_exists:
        logger.info("Write output file header")
        header = ["Cores", "N_train", "N_test", "N_TILES", "N_regressor",
                  "Opt_iter", "Total_time", "Init_time", "Optimization_Time",
                  "Pred_Var_time", "Pred_Full_time", "Predict_time", "N_loop"]
        output_csv_obj.writerow(header)

    # load data_sizes used for multiple iterations of training:
    # start, start+step, start+2*step, ..., end
    start = config["START"]
    end = config["END"]
    step = config["STEP"]

    # runs tests on exponentially increasing number of cores, for linearly
    # increasing data size, for multiple loops (each loop starts with *s)
    for core in range(1, config["N_CORES"]):
        for data_size in range(start, end+step, step):
            for l in range(config["LOOP"]):
                logger.info("*" * 40)
                logger.info(f"Core: {2**core}, Train Size: {data_size}, Loop: {l}")
                gpx_run(config, output_csv_obj, data_size, l, 2**core)


if __name__ == "__main__":
    execute()
