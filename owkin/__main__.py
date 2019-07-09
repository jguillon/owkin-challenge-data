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

from owkin.cli import log_args


def train(args):
    from owkin.ml import train_tiles_classifier, train_subjects_classifier

    logging.info('training:')
    log_args(args)

    if args.tiles_only:
        train_tiles_classifier(data_dir=args.data_dir,
                               model=args.tiles_classifier)
    elif args.subjects_only:
        train_subjects_classifier(data_dir=args.data_dir)
    else:
        train_tiles_classifier(data_dir=args.data_dir)
        train_subjects_classifier(data_dir=args.data_dir)


def test(args):
    logging.info('testing:')
    log_args(args)

    # from .io import load_test_features, load_model
    # logging.info('testing:')
    #
    # logging.info('loading model')
    # model = load_model(filename=args.model)
    #
    # x_test, test_output = load_test_features(args.data_dir)
    #
    # y_test_pred = []
    # y_test_pred_proba = []
    # for i, x in enumerate(x_test):
    #     y_tiles_pred_proba = model.predict_proba(x)[:, 1]
    #     y_tiles_pred = model.predict(x)
    #     y_test_pred_proba.append(y_tiles_pred_proba.max())
    #     y_test_pred.append(y_tiles_pred.max())
    #     logging.debug(
    #             f"{test_output.index[i]} "
    #             f"score = {y_test_pred[i]}, "
    #             f"score = {y_test_pred_proba[i]}")
    #
    # test_output["Target"] = y_test_pred_proba
    # test_output.to_csv(args.data_dir / f"preds_test_{args.model}_proba.csv")
    # test_output["Target"] = y_test_pred
    # test_output.to_csv(args.data_dir / f"preds_test_{args.model}.csv")

    from owkin.ml import build_model, project_as_histogram
    from owkin.io import load_test_images
    from owkin._configuration import HEIGHT, WIDTH
    from keras.applications.resnet50 import preprocess_input
    import numpy as np

    # Model loading
    model = build_model()
    model.load_weights('models/best_transfer_learning.hd5')

    # Data loading
    test_data_generator, test_filenames_df = load_test_images(
            data_dir=Path('data'),
            target_size=(HEIGHT, WIDTH),
            batch_size=8,
            preprocessing_function=preprocess_input
    )

    # Tiles tumoral prediction
    logging.info("predicting the tiles status of the test subjects...")
    y_test_pred = model.predict_generator(test_data_generator,
                                          steps=len(test_data_generator),
                                          verbose=1)

    # Save tiles predictions
    output_file = Path('data/test_output/tiles_predictions.npy')
    logging.info(f"saving predictions in '{output_file}'")
    np.save(output_file, y_test_pred)

    assert len(y_test_pred) == len(test_filenames_df), "the are not corresponding to the list of tiles"

    test_output = test_filenames_df.copy()
    test_output['Target'] = np.zeros(len(test_output))
    test_output['Target'].iloc[:len(y_test_pred)] = y_test_pred[:, 1]

    x_test_2 = test_output.groupby('ID').apply(
            lambda x: project_as_histogram(x['Target']))


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
                                 help="directory where data is stored ("
                                      "default: data/)")
    train_subparser.add_argument("--num_runs", default=3, type=int,
                                 help="number of runs for the cross "
                                      "validation (default: 3)")
    train_subparser.add_argument("--num_splits", default=5, type=int,
                                 help="number of splits for the cross "
                                      "validation (default: 5)")
    train_subparser.add_argument("--tiles_classifier",
                                 default='resnet50',
                                 type=str,
                                 choices=['resnet50', 'random_forest', 'svm'],
                                 help="model for the tiles classification ("
                                      "default: resnet50)")
    train_subparser.add_argument("--subjects_classifier",
                                 default='random_forest_classifier',
                                 type=str,
                                 choices=['random_forest', 'max_proba',
                                          'mean_proba'],
                                 help="model for the subjects classification "
                                      "(default: random_forest)")
    train_subparser.add_argument("--tiles_only", action="store_true",
                                 help="train only the tiles classifier")
    train_subparser.add_argument("--subjects_only", action="store_true",
                                 help="train only the subjects classifier")
    train_subparser.add_argument("--cv", action="store_true",
                                 help="cross-validate")

    test_subparser = subparsers.add_parser('test',
                                           add_help=False,
                                           parents=[parser],
                                           help='test a previously trained '
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

    from owkin._configuration import configure_logger
    configure_logger(debug=args.verbose)

    try:
        args.func(args)
    except KeyboardInterrupt:
        logging.info('\n\ninterruption detected; goodbye!')
        exit(-1)
