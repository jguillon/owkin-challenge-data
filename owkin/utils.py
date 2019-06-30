"""

"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


def load_training_features(data_dir, with_annotated=True):
    """Load the ResNet50 features of the training set.

    Args:
        data_dir (pathlib.Path): directory containing the data tree.
        with_annotated (bool): flag indicating whether to include the features
            and labels of the annotated subjects; in case you already used it
            in a previous training procedure.

    Returns:
        x_train: a list of `n_subjects` numpy arrrays of size
            `(n_tiles, n_features)`.
        train_output (pandas.DataFrame): pandas DataFrame containing list of
        the subjects' ID and
            their associated Target label.

    """
    assert data_dir.is_dir(), f"{data_dir} is not a directory"
    train_dir = data_dir / "train_input" / "resnet_features"
    train_output_filename = data_dir / "train_output.csv"

    # Load training output
    logging.debug(f'reading csv file: {train_output_filename}')
    train_output = pd.read_csv(train_output_filename)

    # Remove annotated subjects
    if not with_annotated:
        train_output = train_output[train_output["ID"].str[-9:] != 'annotated']

    # Load training data
    filenames_train = [train_dir / f"{idx}.npy" for idx in
                       train_output["ID"]]
    for filename in filenames_train:
        assert filename.is_file(), f"{filename} is not a file"

    # Load numpy arrays
    x_train = []
    for f in filenames_train:
        logging.debug(f'loading {f}')
        patient_features = np.load(f)
        x_train.append(patient_features)

    return x_train, train_output


def load_annotated_training_features(data_dir):
    """Load the ResNet50 features of the annotated training set.

    Args:
        data_dir (pathlib.Path): directory containing the data tree.

    Returns:
        x_annot_train (numpy.Array): array of size `n_tiles` x `n_features`.
        y_annot_train (numpy.Array): binary vector of length `n_tiles`.

    """
    assert data_dir.is_dir(), f"{data_dir} is not a directory"
    train_dir = data_dir / "train_input" / "resnet_features"

    train_tiles_annot_filename = data_dir / "train_input" / \
                                 "train_tile_annotations.csv"
    assert train_tiles_annot_filename.is_file(), \
        f"{train_tiles_annot_filename} is not a file"

    logging.debug(f'reading csv file: {train_tiles_annot_filename}')
    train_tiles_annot = pd.read_csv(train_tiles_annot_filename)

    # Get the labels
    y_annot_train = train_tiles_annot["Target"].values

    # Get the filenames
    train_ids_annot = train_tiles_annot.iloc[:, 0].str[:6].unique()
    filenames_train = [train_dir / f"{idx}_annotated.npy" for idx in
                       train_ids_annot]
    for filename in filenames_train:
        assert filename.is_file(), f"{filename} is not a file"

    # Load numpy arrays
    features = []
    for f in filenames_train:
        logging.debug(f'loading {f}')
        patient_features = np.load(f)
        features.append(patient_features)

    x_annot_train = np.concatenate(features)
    assert len(x_annot_train) == len(y_annot_train)

    return x_annot_train, y_annot_train


def load_test_features(data_dir):
    """Load the ResNet50 features of the public test set.

    Args:
        data_dir: (pathlib.Path) directory containing the data tree

    Returns:
        x_test: a list of `n_subjects` numpy arrrays of size
            `(n_tiles, n_features)`.
        test_output: pandas DataFrame containing list of the subjects' ID and
            their associated Target label initialized with zero.

    """
    assert data_dir.is_dir(), f"{data_dir} is not a directory"
    test_dir = data_dir / "test_input" / "resnet_features"

    # Get the test filenames
    filenames_test = sorted(test_dir.glob("*.npy"))
    for filename in filenames_test:
        assert filename.is_file(), f"{filename} is not a file"
    ids_test = [f.stem for f in filenames_test]

    ids_number_test = [i.split("ID_")[1] for i in ids_test]
    test_output = pd.DataFrame(
            {"ID": ids_number_test, "Target": np.zeros(len(ids_test))})
    test_output.set_index("ID", inplace=True)

    # Load numpy arrays
    x_test = []
    for f in filenames_test:
        logging.debug(f'loading {f}')
        x_test.append(np.load(f))


def load_resnet_features(data_dir):
    """Load the ResNet50 feature vectors of all the annotated tiles with their
    corresponding target.

    Args:
        data_dir: directory where the predefined data tree is stored

    Returns:
        x_train: n x p array, where n is the number of tiles and p the number of
            features.
        y_train: n x 1 vector, containing zeros for normal, ones for metastatic.

    """

    # Load the data
    assert data_dir.is_dir(), f"{data_dir} is not a directory"

    train_dir = data_dir / "train_input" / "resnet_features"
    test_dir = data_dir / "test_input" / "resnet_features"
    train_tiles_annot_filename = data_dir / "train_input" / \
                                 "train_tile_annotations.csv"
    train_output_filename = data_dir / "train_output.csv"

    logging.debug(f'reading csv file: {train_tiles_annot_filename}')
    train_tiles_annot = pd.read_csv(train_tiles_annot_filename)

    # Get the labels
    y_train = train_tiles_annot["Target"].values

    # Get the filenames
    train_ids_annot = train_tiles_annot.iloc[:, 0].str[:6].unique()
    filenames_train = [train_dir / f"{idx}_annotated.npy" for idx in
                       train_ids_annot]
    for filename in filenames_train:
        assert filename.is_file(), f"{filename} is not a file"

    # Load numpy arrays
    features = []
    for f in filenames_train:
        logging.debug(f'loading {f}')
        patient_features = np.load(f)
        features.append(patient_features)

    x_train = np.concatenate(features)
    assert len(x_train) == len(y_train)

    # Load training output
    logging.debug(f'reading csv file: {train_output_filename}')
    train_output = pd.read_csv(train_output_filename)

    # Load training data
    train_output_na = train_output[train_output["ID"].str[-9:] != 'annotated']
    filenames_train_na = [train_dir / f"{idx}.npy" for idx in
                          train_output_na["ID"]]
    for filename in filenames_train_na:
        assert filename.is_file(), f"{filename} is not a file"

    # Load numpy arrays
    x_train_na = []
    for f in filenames_train_na:
        logging.debug(f'loading {f}')
        patient_features = np.load(f)
        x_train_na.append(patient_features)

    return x_train, y_train, x_train_na, train_output_na


def load_training_images():
    pass


def load_test_images():
    pass


def submit_test_output(test_output, file_id, output_dir=Path('outputs/')):
    return x_test, test_output

    """Submit the test output by simply writing it to a .csv file.

    Args:
        test_output: pandas DataFrame containing list of the subjects' ID and
            their associated Target label.

    Returns: None.

    """
    filename = output_dir / f"preds_test_{file_id}.csv"
    logging.debug(f'writing to {filename}')
    test_output.to_csv(filename)
