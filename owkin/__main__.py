#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import argparse
import logging
from pathlib import Path

import argcomplete
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

import owkin.utils as utils

def train(args):
    logging.info('training:')
    for k, v in vars(args).items():
        if k not in ['func', 'verbose']:
            logging.info(f'  {k}={v}')
    logging.info('')

    # Cross-validation of the tiles classifier
    # ========================================
    #
    # The goal here is to classify whether a tile contains metastasis or not.

    # Load the data
    logging.info(f"loading data from {args.data_dir}")
    x_train, y_train = utils.load_annotated_training_features(args.data_dir)
    x_train_na, train_output_na = utils.load_training_features(args.data_dir,
                                                               with_annotated=False)

    # Preproc the data
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_train = x_train[:,3:]

    if True:
        # Multiple cross validations on the training set
        logging.info("")
        logging.info("cross-validation:")
        aucs = []
        estimator = None
        for seed in range(args.num_runs):
            logging.info(f"  run {seed + 1}/{args.num_runs}:")

            # Use logistic regression with L2 penalty
            estimators = {
                'random_forest_classifier': RandomForestClassifier(
                    n_estimators=20),
                'svm'                     : SVC(),
            }
            estimator = estimators[args.model]
            cv = StratifiedKFold(n_splits=args.num_splits,
                                 shuffle=True,
                                 random_state=seed)

            # Cross validation on the training set
            y_train_pred = cross_val_predict(estimator,
                                             X=x_train,
                                             y=y_train,
                                             n_jobs=-1,
                                             method='predict_proba',
                                             cv=cv,  # scoring="roc_auc",
                                             verbose=3 if args.verbose else 0)

            auc = roc_auc_score(y_train, y_train_pred[:, 1])
            logging.debug(f"    auc={auc}")

            aucs.append(auc)

        aucs = np.array(aucs)
        logging.info("  auc: mean {}, std {}".format(aucs.mean(), aucs.std()))

    # Validation of the subject classifier
    # ====================================

    # Training
    logging.info(
        "training the tiles classifier on the entire annotated dataset")
    estimators = {
        'random_forest_classifier': RandomForestClassifier(n_estimators=20),
    }
    estimator = estimators[args.model]
    estimator.fit(x_train, y_train)

    # weak validation on all the non-annotated tiles:
    y_true = train_output_na['Target'].values
    y_pred = []
    y_pred_proba = []
    for i, x in enumerate(x_train_na):
        y_tiles_pred_proba = estimator.predict_proba(x)[:, 1]
        y_tiles_pred = estimator.predict(x)
        y_pred_proba.append(y_tiles_pred_proba.mean())
        y_pred.append(y_tiles_pred.mean())
        logging.debug(
            f"{train_output_na['ID'].iloc[i]} "
            f"score = {y_pred[i]}, "
            f"score = {y_pred_proba[i]}, "
            f"true = {y_true[i]}")

    final_auc_proba = roc_auc_score(y_true, y_pred_proba)
    final_auc = roc_auc_score(y_true, y_pred)
    logging.info(f"final auc proba: {final_auc_proba}")
    logging.info(f"final auc: {final_auc}")

    # Test
    x_test, test_output = utils.load_test_features(args.data_dir)

    y_test_pred = []
    y_test_pred_proba = []
    for i, x in enumerate(x_test):
        y_tiles_pred_proba = estimator.predict_proba(x)[:, 1]
        y_tiles_pred = estimator.predict(x)
        y_test_pred_proba.append(y_tiles_pred_proba.mean())
        y_test_pred.append(y_tiles_pred.mean())
        logging.debug(
            f"{test_output.index[i]} "
            f"score = {y_test_pred[i]}, "
            f"score = {y_test_pred_proba[i]}")

    test_output["Target"] = y_test_pred_proba
    test_output.to_csv(args.data_dir / f"preds_test_{args.model}_proba.csv")
    test_output["Target"] = y_test_pred
    test_output.to_csv(args.data_dir / f"preds_test_{args.model}.csv")


def test(args):
    logging.info('testing:')

    # load estimator:

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

    parser = argparse.ArgumentParser(prog='owkin')
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
    train_subparser.add_argument("--model", default='svm', type=str,
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
