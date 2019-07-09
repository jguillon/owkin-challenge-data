"""

"""

import logging


def project_as_histogram(x, bins=20):
    """Project a probabilities vector to an histogram representation.

    Args:
        x: a vector of probabilities of length `n_tiles`
        bins (int): size of the histogram output vector

    Returns: the histogram vector of length `bins`, containing the proportion
    of
        values

    """
    import numpy as np

    counts, _ = np.histogram(x, bins=bins, range=(0, 1))
    return counts / sum(counts)


def build_model(hidden_layer_size=32, output_layer_size=2, pooling='avg'):
    """

    Returns:

    """
    from ._configuration import HEIGHT, WIDTH, CHANNELS
    from keras.applications import ResNet50
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential([
        ResNet50(weights='imagenet', include_top=False,
                 input_shape=(HEIGHT, WIDTH, CHANNELS), pooling=pooling),
        Dense(hidden_layer_size, activation='relu'),
        Dense(output_layer_size, activation='softmax')
    ])

    # we do not train the ResNet50 layer since it is already pre-trained on
    # the ImageNet dataset
    model.layers[0].trainable = False

    return model


def train_tiles_classifier(data_dir, cross_val=True, save=True, num_runs=10, num_splits=10,
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
        for seed in range(num_runs):
            logging.info(f"  run {seed + 1}/{num_runs}:")

            # Model initialisation
            estimators = {
                'random_forest': RandomForestClassifier(n_estimators=20),
                'svm'          : SVC(),
            }
            estimator = estimators[model]
            cv = StratifiedKFold(n_splits=num_splits, shuffle=True,
                                 random_state=seed)

            # Cross validation on the training set
            auc = cross_val_score(estimator, X=x_train, y=y_train, n_jobs=-1,
                                  cv=cv, scoring="roc_auc", verbose=verbose)
            logging.debug(f"    auc={auc}")
            aucs.append(auc)

        aucs = np.array(aucs)
        logging.info("  auc: mean {}, std {}".format(aucs.mean(), aucs.std()))
    else:
        # Single training on the whole dataset
        logging.info("training the tiles classifier on the entire annotated "
                     "dataset")
        estimators = {
            'random_forest': RandomForestClassifier(n_estimators=20),
            'svm'          : SVC(),
        }
        estimator = estimators[model]
        estimator.fit(x_train, y_train)

        if save:
            # Model saving
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
