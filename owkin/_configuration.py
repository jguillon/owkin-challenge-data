"""

"""

import logging
import sys

from colorama import Fore, Style

HEIGHT = 224
WIDTH = 224
CHANNELS = 3

def configure_logger(debug=False):
    """

    Args:
        args: parsed arguments as given by the `parse_known_args()` function.

    Returns: None

    """
    # Set logging level and format
    logger = logging.getLogger()
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Redirect lvl > INFO to stdout
    out_handler_debug = logging.StreamHandler(sys.stdout)
    out_handler_debug.setLevel(logging.DEBUG)
    out_handler_debug.setFormatter(
            logging.Formatter(
                    f'{Fore.LIGHTGREEN_EX}%(message)s{Style.RESET_ALL}'))
    out_handler_debug.addFilter(lambda record: record.levelno == logging.DEBUG)
    logger.addHandler(out_handler_debug)

    out_handler_info = logging.StreamHandler(sys.stdout)
    out_handler_info.setLevel(logging.INFO)
    out_handler_info.setFormatter(
            logging.Formatter('%(message)s'))
    out_handler_info.addFilter(lambda record: record.levelno == logging.INFO)
    logger.addHandler(out_handler_info)

    err_handler_warn = logging.StreamHandler(sys.stderr)
    err_handler_warn.setLevel(logging.WARNING)
    err_handler_warn.setFormatter(
            logging.Formatter(
                    f'{Fore.YELLOW}warning: %(message)s{Style.RESET_ALL}'))
    err_handler_warn.addFilter(lambda record: record.levelno == logging.WARNING)
    logger.addHandler(err_handler_warn)

    err_handler_error = logging.StreamHandler(sys.stderr)
    err_handler_error.setLevel(logging.ERROR)
    err_handler_error.setFormatter(
            logging.Formatter(
                    f'{Fore.RED}error: %(message)s{Style.RESET_ALL}'))
    err_handler_error.addFilter(lambda record: record.levelno == logging.ERROR)
    logger.addHandler(err_handler_error)
