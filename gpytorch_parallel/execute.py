import time
import logging
from csv import writer
import torch
import gpytorch

from config import get_config
from gpytorch_logger import setup_logging
from utils import load_data, ExactGPModel, train, predict, calculate_error

logger = logging.getLogger()
log_filename = "./gpytorch_logs.log"


def gpytorch_run(config, output_csv_obj, size_train, l):
    """
    Run the Gaussian process regression pipeline.

    Args:
        config (dict): Configuration parameters for the pipeline.
        output_csv_obj (csv.writer): CSV writer object for writing output data.
        size_train (int): Size of the training dataset.
        l (int): Loop index.
    """
    total_t = time.time()
    
    X_train, Y_train, X_test, Y_test = load_data(
        train_in_path=config["train_in_file"],
        train_out_path=config["train_out_file"],
        test_in_path=config["test_in_file"],
        test_out_path=config["test_out_file"],
        size_train=size_train,
        size_test=config["N_TEST"],
        n_regressors=config["N_REG"],
    )

    logger.info("Finished loading the data.")

    init_t = time.time()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = 0.1
    model = ExactGPModel(X_train, Y_train, likelihood)
    init_t = time.time() - init_t
    logger.info("Initialized model.")
    
    train_t = time.time()
    train(model, likelihood, X_train, Y_train)
    train_t = time.time() - train_t
    logger.info("Trained model.")

    pred_t = time.time()
    f_pred, f_var, y_pred, y_var = predict(model, likelihood, X_test)
    pred_t = time.time() - pred_t
    logger.info("Finished making predictions.") 
    
    TOTAL_TIME = time.time() - total_t
    INIT_TIME = init_t
    TRAIN_TIME = train_t
    PREDICTION_TIME = pred_t
    ERROR = calculate_error(Y_test, y_pred).detach().cpu().numpy()
    
    row_data = [config["N_CORES"], size_train, config["N_TEST"], config["N_REG"], 
                TOTAL_TIME, INIT_TIME, TRAIN_TIME, PREDICTION_TIME, ERROR, l]
    output_csv_obj.writerow(row_data)
    
    logger.info("Completed iteration.")
    

def execute():
    """
    This function performs following steps:
        - Set up logging.
        - Load configuration file.
        - Write header for the output CSV file.
        - Set up PyTorch configurations based on the loaded config.
        - Iterate through different training sizes and for each training size
        loop for a specified amount of times while executing `gpytorch_run` function.
    """
    setup_logging(log_filename, True, logger)
    logger.info("\n")
    logger.info("-" * 40)
    logger.info("Load config file.")
    config = get_config()
    output_file = open("./output.csv", "a", newline="")
    output_csv_obj = writer(output_file)
    
    logger.info("Write output file header")
    header = ["Cores", "N_train", "N_test", "N_regressor", "Total_time",
         "Train_time", "Optimization_Time", "Predict_time", "Error", "N_loop"]
    output_csv_obj.writerow(header)

    start = config["START"]
    end = config["END"]
    step = config["STEP"]
    torch.set_num_threads(config["N_CORES"])
    if config["PRECISION"] == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float64)

    for i in range(start, end+step, step):
        for l in range(config["LOOP"]):
            logger.info("*" * 40)
            logger.info(f"Train Size: {i}, Loop: {l}")
            gpytorch_run(config, output_csv_obj, i, l)

    output_file.close()
    
    logger.info("Completed the program.")


if __name__ == "__main__":
    execute()
