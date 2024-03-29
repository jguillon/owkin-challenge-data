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
    train_output_filename = data_dir / "training_output.csv"

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

    return x_test, test_output


def _load_image_generator(batch_size, target_size, image_dir, preprocessing_function=None, with_annotated=True):
    """
    """
    from keras_preprocessing.image import \
        ImageDataGenerator  # custom fix from some guy on github
    
    # Collect the images filename
    logging.info(f"listing {image_dir}...")
    filenames_df = pd.DataFrame(columns=['ID', 'Filename'])
    subject_dirs = sorted(image_dir.glob("ID_*"))
    if not with_annotated:
        subject_dirs = [d for d in subject_dirs if d.stem[6:]!='_annotated']
    logging.info(f"the folder contains {len(subject_dirs)} subjects")
    assert len(subject_dirs) > 0, f"folder {image_dir} is empty"
    for subject_dir in subject_dirs:
        if subject_dir.is_dir():
            subject_id = subject_dir.stem[3:]
            logging.info(f"  extracting subjects {subject_id}'s image filenames")
            subject_images = sorted(subject_dir.glob("*.jpg"))
            for image in subject_images:
                assert image.is_file(), f"{image} is not a file"
                logging.debug(f"    {image}")
                filenames_df = filenames_df.append({
                    'ID'      : subject_id,
                    'Filename': 'ID_' + subject_id + '/' + image.stem + image.suffix
                }, ignore_index=True)

    # Create the image generator
    image_generator = ImageDataGenerator(
            preprocessing_function=preprocessing_function)
    image_iterator = image_generator.flow_from_dataframe(
            dataframe=filenames_df, x_col='Filename',
            directory=image_dir,
            target_size=target_size,
            batch_size=batch_size,
            shuffle=False,
            class_mode=None
    )
    
    return image_iterator, filenames_df


def load_training_images(batch_size, target_size, data_dir=Path('data/'),
                     preprocessing_function=None, with_annotated=True):
    """

    """
    
    train_images_dir = data_dir / "train_input" / "images"
    
    train_image_iterator, train_filenames_df = _load_image_generator(
        image_dir=train_images_dir,
        batch_size=batch_size,
        target_size=target_size,
        preprocessing_function=preprocessing_function,
        with_annotated=with_annotated
    )
    
    train_output_filename = data_dir / "training_output.csv"

    # Load training output
    logging.info(f'reading csv file: {train_output_filename}')
    train_output = pd.read_csv(train_output_filename)
    train_output = train_output.set_index('ID')
    
    # Filter dataframe with loaded subjects
    ids = pd.to_numeric(train_filenames_df['ID'].str[:3].unique())
    train_output = train_output[train_output.index.isin(ids)]
    
    return train_image_iterator, train_filenames_df, train_output
    

def load_test_images(batch_size, target_size, data_dir=Path('data/'),
                     preprocessing_function=None):
    """

    """
    
    return _load_image_generator(
        image_dir=data_dir / "test_input" / "images",
        batch_size=batch_size,
        target_size=target_size,
        preprocessing_function=preprocessing_function,
    )
    

def submit_test_output(test_output, file_id, output_dir=Path('outputs/')):
    """Submit the test output by simply writing it to a .csv file.

    Args:
        test_output: pandas DataFrame containing list of the subjects' ID and
            their associated Target label.

    Returns: None.

    """
    filename = output_dir / f"preds_test_{file_id}.csv"
    logging.debug(f'writing to {filename}')
    test_output.to_csv(filename)


def save_model(model, filename, output_dir):
    """Save a sklearn estimator or a keras model.

    Args:
        model: sklearn estimator or keras model.
        filename: stem of the output file (i.e. without extension).

    Returns: None.

    """
    from sklearn.externals import joblib
    joblib.dump(model, f'{filename}.pkl')

def load_model(filename):
    """Load a previously save model.

    Notes: The file has to have been saved using the `save_model` function.

    Args:
        filename: stem of the file to load.

    Returns: The loaded model.

    """
    from sklearn.externals import joblib
    # Load the pickle file
    model = joblib.load(f'{filename}.pkl')

    return model