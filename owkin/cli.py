"""

"""
import logging


def log_args(args):
    for k, v in vars(args).items():
        if k not in ['func', 'verbose']:
            logging.info(f'  {k}={v}')
    logging.info('')
