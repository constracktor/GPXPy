import logging
from os import environ

log = logging.getLogger(__name__)


def log_to_console(log):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(get_formatter())
    log.addHandler(console_handler)


def get_formatter(format=None):
    if not format:
        format = (
            environ.get("env", "")
            + "%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s"
        )
        return logging.Formatter(format)


def setup_logging(log_filename, console_log, logger):
    if not logger.handlers:
        formatter = get_formatter()
        level = logging.INFO
        if log_filename:

            handler = logging.FileHandler(log_filename)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.setLevel(level)
        """if verbose:
            logger.setLever(logging.DEBUG)"""
        if console_log:
            log_to_console(logger)
