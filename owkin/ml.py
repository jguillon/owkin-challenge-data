"""

"""

import logging


def train_tiles_classifier(data_dir, cross_val=True, num_runs=10, num_splits=10,
                           verbose=3, model='random_forest',
                           filename='tiles_classifier'):
    """Cross-validation and training of the tiles classifier.

    The goal here is to classify whether a tile contains metastasis or not.

    Args:
        data_dir (Path):
        cv (bool):
        num_runs (int):
        num_splits (int):
        verbose (int):
        model (str):

    Returns:

    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from .io import load_annotated_training_features, save_model
    import numpy as np

    # Load the data
    logging.info(f"loading data from {data_dir}")
    x_train, y_train = load_annotated_training_features(data_dir)

    # Preproc the data
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_train = x_train[:,3:]

    if cross_val:
        # Multiple cross validations on the training set
        logging.info("")
        logging.info("cross-validation:")
        aucs = []
        estimator = None
        for seed in range(num_runs):
            logging.info(f"  run {seed + 1}/{num_runs}:")

            # Use logistic regression with L2 penalty
            estimators = {
                'random_forest': RandomForestClassifier(
                        n_estimators=20),
                'svm'          : SVC(),
            }
            estimator = estimators[model]
            cv = StratifiedKFold(n_splits=num_splits,
                                 shuffle=True,
                                 random_state=seed)

            # Cross validation on the training set
            auc = cross_val_score(estimator,
                                  X=x_train,
                                  y=y_train,
                                  n_jobs=-1,
                                  # method='predict_proba',
                                  cv=cv,
                                  scoring="roc_auc",
                                  verbose=verbose)

            # auc = roc_auc_score(y_train, y_train_pred[:, 1])
            logging.debug(f"    auc={auc}")

            aucs.append(auc)

        aucs = np.array(aucs)
        logging.info("  auc: mean {}, std {}".format(aucs.mean(), aucs.std()))
    else:
        # Training
        logging.info("training the tiles classifier on the entire annotated "
                     "dataset")
        estimators = {
            'random_forest': RandomForestClassifier(n_estimators=20),
            'svm'          : SVC(),
        }
        estimator = estimators[model]
        estimator.fit(x_train, y_train)

        save_model(estimator, filename=model)


def train_subjects_classifier(data_dir, cv=True, num_runs=10, num_splits=10,
                              verbose=3, model='random_forest',
                              filename='tiles_classifier'):
    """

    Args:
        data_dir:
        cv:
        num_runs:
        num_splits:
        verbose:
        model:
        filename:

    Returns:

    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from .io import load_training_features

    # Load the data
    logging.info(f"loading data from {data_dir}")
    x_train, train_output = load_training_features(data_dir,
                                                   with_annotated=False)

    estimators = {
        'random_forest': RandomForestClassifier(
                n_estimators=20),
        'svm'          : SVC(),
    }
    estimator = estimators[model]

    for i, x in enumerate(x_train):
        y_tiles_pred_proba = estimator.predict_proba(x)[:, 1]
        y_tiles_pred = estimator.predict(x)
        y_test_pred_proba.append(y_tiles_pred_proba.max())
        y_test_pred.append(y_tiles_pred.max())
        logging.debug(
                f"{test_output.index[i]} "
                f"score = {y_test_pred[i]}, "
                f"score = {y_test_pred_proba[i]}")
