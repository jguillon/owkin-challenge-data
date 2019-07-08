#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main entrance point of the commandline tool.

Warning: Never put any unnecessary imports at the top of this file or the CL
will take much more time to respond.
"""

import argparse
import logging
from pathlib import Path

import argcomplete

from .cli import log_args


def train(args):
    from .ml import train_tiles_classifier, train_subjects_classifier
    from sklearn.ensemble import RandomForestClassifier

    logging.info('training:')
    log_args(args)

    train_tiles_classifier()

    train_subjects_classifier()


def test(args):
    from .io import load_test_features, load_model
    logging.info('testing:')

    logging.info('loading model')
    model = load_model(filename=args.model)

    x_test, test_output = load_test_features(args.data_dir)

    y_test_pred = []
    y_test_pred_proba = []
    for i, x in enumerate(x_test):
        y_tiles_pred_proba = model.predict_proba(x)[:, 1]
        y_tiles_pred = model.predict(x)
        y_test_pred_proba.append(y_tiles_pred_proba.max())
        y_test_pred.append(y_tiles_pred.max())
        logging.debug(
                f"{test_output.index[i]} "
                f"score = {y_test_pred[i]}, "
                f"score = {y_test_pred_proba[i]}")

    test_output["Target"] = y_test_pred_proba
    test_output.to_csv(args.data_dir / f"preds_test_{args.model}_proba.csv")
    test_output["Target"] = y_test_pred
    test_output.to_csv(args.data_dir / f"preds_test_{args.model}.csv")

    # # Test
    # y_test_pred = []
    # y_test_pred_proba = []
    # for i, x in enumerate(x_test):
    #     y_tiles_pred_proba = estimator.predict_proba(x)[:, 1]
    #     y_tiles_pred = estimator.predict(x)
    #     y_test_pred_proba.append(y_tiles_pred_proba.mean())
    #     y_test_pred.append(y_tiles_pred.mean())
    #     logging.debug(
    #         f"{test_output.index[i]} score = {y_test_pred[i]}, score = {
    #         y_test_pred_proba[i]}")


def submit(args):
    logging.info('submitting:')

    # test_output["Target"] = y_test_pred_proba
    # test_output.to_csv(args.data_dir / f"preds_test_{args.model}_proba.csv")
    # test_output["Target"] = y_test_pred
    # test_output.to_csv(args.data_dir / f"preds_test_{args.model}.csv")


if __name__ == '__main__':
    # Configure arguments parsers
    # ===========================

    parser = argparse.ArgumentParser(prog='owkin',
                                     description='This command line tool '
                                                 'allows ' \
                                                 'to train or test any of the '
                                                 '' \
                                                 'developed models from a '
                                                 'shell ' \
                                                 'console.')
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    subparsers = parser.add_subparsers(title='commands')

    # Add sub-commands to acct command line
    train_subparser = subparsers.add_parser('train',
                                            add_help=False,
                                            parents=[parser],
                                            help='cross-validate and train')
    train_subparser.add_argument("--data_dir", type=Path,
                                 default='data/',
                                 help="directory where data is stored")
    train_subparser.add_argument("--num_runs", default=3, type=int,
                                 help="number of runs for the cross validation")
    train_subparser.add_argument("--num_splits", default=5, type=int,
                                 help="number of splits for the cross "
                                      "validation")
    train_subparser.add_argument("--model", default='random_forest_classifier',
                                 type=str,
                                 choices=['random_forest_classifier', 'svm'],
                                 help="number of splits for the cross "
                                      "validation")

    test_subparser = subparsers.add_parser('test',
                                           add_help=False,
                                           parents=[parser],
                                           help='test a previously train '
                                                'estimator on the public test'
                                                ' set')

    # Link the default function to each sub-command
    train_subparser.set_defaults(func=train)
    test_subparser.set_defaults(func=test)
    # submit_subparser.set_defaults(func=submit)

    # Parse and run
    # =============

    args = None
    unknown_args = None
    try:
        argcomplete.autocomplete(parser)
        args, unknown_args = parser.parse_known_args()
    except SystemExit:
        exit(-1)
    except Exception:
        parser.print_help()
        exit(-1)

    if unknown_args:
        raise ValueError('Unknown flag detected: %s' % unknown_args)

    if args is None or hasattr(args, 'func') is False:
        parser.print_help()
        exit(-1)

    if args.verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    try:
        args.func(args)
    except KeyboardInterrupt:
        logging.info('\n\ninterruption detected; goodbye!')
        exit(-1)
